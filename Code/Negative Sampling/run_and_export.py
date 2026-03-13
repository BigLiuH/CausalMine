import torch
import numpy as np
import os
from util.conf import ModelConf
from data.loader import FileIO
from model.graph.CNSDiff import CNSDiff
from conf.params import args


os.chdir(os.path.dirname(__file__))
def run_and_export():
    # 1. Load Configuration
    print("Loading configuration...")
    # Use absolute path to avoid file not found errors
    current_dir = os.path.dirname(os.path.abspath(__file__))
    conf_path = os.path.join(current_dir, 'conf', 'CNSDiff.yaml')
    conf = ModelConf(conf_path)
    
    # 2. Load Data
    print("Loading data...")
    training_data = FileIO.load_data_set(conf['training.set'], conf['model']['type'])
    test_data = FileIO.load_data_set(conf['test.set'], conf['model']['type'])
    
    # 3. Initialize Model
    print("Initializing CNSDiff model...")
    model = CNSDiff(conf, training_data, test_data)
    
    # 4. Train the model
    print("Starting training...")
    # execute() runs build(), train(), test(), evaluate()
    model.execute()
    
    # 5. Generate Hard Negatives
    print("\n" + "="*50)
    print("Training complete. Starting hard negative generation...")
    print("="*50)
    
    # We need to generate negatives for each user (or interaction).
    # Let's generate for all users in the training set.
    # We need to pass user_emb and pos_item_emb.
    # If we want generic negatives for a user, what should pos_item_emb be?
    # The generate_hard_negatives method uses pos_item_emb as the starting point for diffusion (or condition).
    # If we want to predict "future" items, we might not have a pos_item_emb.
    # However, the method signature requires it.
    # Let's assume we want to generate negatives for the *existing* positive interactions 
    # (to see what the model thinks are "hard" alternatives).
    
    # Alternatively, if we just want to sample items for a user, we might need a different method 
    # that starts from noise or a "prototype" item.
    # Given the code uses `pos_item_emb`, let's iterate over the training interactions.
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    # Get all embeddings
    # CNSDiff class does not have forward(), but its internal model (LGCN_Encoder) does.
    # Or we can use the stored embeddings if available.
    # In CNSDiff.train(), self.user_emb and self.item_emb are updated.
    # Let's use those directly.
    
    if hasattr(model, 'user_emb') and hasattr(model, 'item_emb'):
        user_emb_all = model.user_emb
        item_emb_all = model.item_emb
    else:
        # Fallback: call the encoder model directly
        user_emb_all, item_emb_all = model.model()
    
    # ---------------------------------------------------------
    # 生成负样本的方法说明（中文）
    # ---------------------------------------------------------
    # 流程概述：
    # 1) 对每个用户的每个正交互，使用 model.generate_hard_negatives(user_emb, pos_item_emb)
    #    生成对应的“负向向量”（即模型认为的困难负样本的向量表示）。
    # 2) 将生成的负向向量与所有物品向量都进行 L2 归一化（cosine 相似度计算）。
    # 3) 通过矩阵乘法计算负向量与所有物品的相似度得分（越大越相似）。
    # 4) 将该用户的所有正例在相似度表中屏蔽（设为 -inf），避免采到已交互物品。
    # 5) 对每个正交互取 top-K 候选（K = num_pos + 20），使用贪心策略为每个交互挑选
    #    一个未被本用户其它交互占用且不是用户本身或该正例的负样本，保证同一用户负样本唯一。
    # 6) 若 top-K 中未找到合法候选，回退尝试候选列表中任一合法项；若仍无则跳过该交互。
    # 7) 将最终的 (user, pos_item, neg_item) 三元组保存到文件。
    #
    # 关键点：
    # - 负样本向量由模型生成（不是简单随机或基于流行度），因此被称为“hard negatives”。
    # - 通过屏蔽正例并贪心去重，尽量保证每个正例对应一个有效且不重复的负样本。
    # - 若需更严格/多样的负采样策略，可调整 K 或改为更复杂的分配算法。
    
    # ---------------------------------------------------------
    # Generate Negatives and Find Neighbors (User-by-User)
    # ---------------------------------------------------------
    print("Generating negatives and finding neighbors (User-by-User strategy)...")
    
    model.MLP.eval()
    model.Diffusion.eval()
    
    # Prepare Item Embeddings for Search
    import torch.nn.functional as F
    if not isinstance(item_emb_all, torch.Tensor):
        item_emb_all = torch.tensor(item_emb_all)
    item_emb_all = item_emb_all.to(device)
    items_norm = F.normalize(item_emb_all, p=2, dim=1)
    
    final_triplets = [] # List of (User_ID, Pos_Item_ID, Neg_Item_ID)
    
    # Iterate over each user in the dataset
    # model.data.training_set_u is {user_str: {item_str: rating}}
    
    user_list = list(model.data.training_set_u.keys())
    total_users = len(user_list)
    
    with torch.no_grad():
        for i, user_str in enumerate(user_list):
            if i % 100 == 0:
                print(f"Processed {i}/{total_users} users...")
                
            # Get internal User ID
            u_id_internal = model.data.user[user_str]
            
            # Get all positive items for this user
            pos_items_str = list(model.data.training_set_u[user_str].keys())
            pos_items_internal = [model.data.item[item] for item in pos_items_str]
            
            num_pos = len(pos_items_internal)
            if num_pos == 0:
                continue
                
            # Prepare tensors for this user's batch
            # User Emb: Repeat for each positive item
            batch_u = torch.tensor([u_id_internal] * num_pos, dtype=torch.long).to(device)
            batch_i = torch.tensor(pos_items_internal, dtype=torch.long).to(device)
            
            batch_user_emb = user_emb_all[batch_u]
            batch_pos_item_emb = item_emb_all[batch_i]
            
            # Generate Negative Vectors
            # Shape: [Num_Pos, Emb_Size]
            neg_emb = model.generate_hard_negatives(batch_user_emb, batch_pos_item_emb)

            # ---------------------------------------------------------
            # 相似度计算说明：
            # 这里采用了【余弦相似度 (Cosine Similarity)】。
            # 原理：Cosine(A, B) = Dot(A, B) / (||A|| * ||B||)
            # 实现：先使用 F.normalize 对向量进行 L2 归一化（模长变为1），
            # 此时 ||A|| = 1, ||B|| = 1，公式简化为 Cosine(A, B) = Dot(A, B)。
            # ---------------------------------------------------------
            neg_norm = F.normalize(neg_emb, p=2, dim=1)
            
            # Compute Similarity to ALL items
            # 通过矩阵乘法批量计算点积
            # Shape: [Num_Pos, Num_Total_Items]
            sim_scores = torch.matmul(neg_norm, items_norm.t())
            
            # Mask Positive Items (Set score to -inf)
            # We mask ALL positive items for this user, for ALL generated vectors
            # This ensures we don't recommend ANY item the user has already interacted with
            mask_indices = torch.tensor(pos_items_internal, dtype=torch.long).to(device)
            sim_scores[:, mask_indices] = -1e9
            
            # Select Unique Negatives
            # We need to select 'num_pos' unique items from the top candidates
            # Strategy: Flatten the scores or iterate greedily?
            # Simple Greedy: For each interaction, pick best available that hasn't been picked
            
            used_negatives = set()
            user_results = []
            
            # Get Top-K candidates for each interaction to have a buffer
            # K = num_pos + 10 (just to be safe)
            K = min(num_pos + 20, model.data.item_num)
            topk_scores, topk_indices = torch.topk(sim_scores, k=K, dim=1)
            
            topk_indices = topk_indices.cpu().numpy()
            
            for idx in range(num_pos):
                # Try to find a valid negative for interaction 'idx'
                candidates = topk_indices[idx]
                found = False
                for cand in candidates:
                    # Check candidate is valid before using
                    cand_str = model.data.id2item[cand]
                    # Skip if: 1) already used for this user, 2) equals user ID, 3) equals positive item
                    if cand not in used_negatives and cand_str != user_str and cand_str != pos_items_str[idx]:
                        neg_item_internal = cand
                        used_negatives.add(cand)
                        found = True
                        break
                
                if not found:
                    # Fallback: find ANY valid candidate (not equal to user_str or pos_item_str)
                    for cand in candidates:
                        cand_str = model.data.id2item[cand]
                        if cand_str != user_str and cand_str != pos_items_str[idx]:
                            neg_item_internal = cand
                            found = True
                            break
                
                # Skip this interaction if no valid negative found
                if not found:
                    continue
                
                # Map back to string IDs
                pos_item_str = pos_items_str[idx]
                neg_item_str = model.data.id2item[neg_item_internal]

                final_triplets.append((user_str, pos_item_str, neg_item_str))


    # ---------------------------------------------------------
    # Save User-Positive-Negative Item triplets
    # ---------------------------------------------------------
    print("Saving User-Positive-Negative Item triplets...")
    pairs_path = './dataset9905/hard_negative_pairs.txt'
    
    with open(pairs_path, 'w') as f:
        for u_str, p_str, n_str in final_triplets:
            f.write(f"{u_str},{p_str},{n_str}\n")
            
    print(f"Successfully saved {len(final_triplets)} triplets to {pairs_path}")
    
    # Optional: Save a few examples to verify
    print("Example mapping (First 5):")
    for i in range(min(5, len(final_triplets))):
        print(f"Triplet: {final_triplets[i]}")

if __name__ == '__main__':
    run_and_export()
