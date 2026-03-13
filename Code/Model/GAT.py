import torch
from torch import nn
from torch_geometric.nn.conv import GATConv
from cbam import CBAM
# -------------------------- ECA模块定义 --------------------------
class ECA(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, use_2d: bool = False):
        super().__init__()
        self.use_2d = use_2d
        self.avg_pool = nn.AdaptiveAvgPool2d(1) if use_2d else nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                             padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, *spatial_dims = x.shape
        if self.use_2d:
            y = self.avg_pool(x).view(b, d, 1)
        else:
            y = self.avg_pool(x)
        y = y.transpose(1, 2)
        y = self.conv(y)
        y = y.transpose(1, 2)
        weights = self.sigmoid(y)
        if self.use_2d:
            return x * weights.view(b, d, 1, 1)
        else:
            return x * weights

# -------------------------- 完整GATModel类 --------------------------
class GATModel(nn.Module):
    def __init__(self, args, mv):
        super(GATModel, self).__init__()
        self.mv = mv
        self.args = args

        # GAT层定义（保持不变）
        self.gat_x1_f = GATConv(self.args.fm, self.args.fm, heads=8, concat=False)# circ（avg）
        self.gat_x2_f = GATConv(self.args.fm, self.args.fm, heads=8, concat=False)# （max）

        self.globalAvgPool_x = nn.AvgPool2d((self.args.fm, self.args.miRNA_number), (1, 1))
        self.globalAvgPool_y = nn.AvgPool2d((self.args.fd, self.args.disease_number), (1, 1))

        # -------------------------- 完整mv分支（所有分支均已替换） --------------------------

        # self.ecax = ECA(channels=1 * self.args.gcn_layers, kernel_size=5, use_2d=True)
        self.cbamx = CBAM(1 * self.args.gcn_layers, 5, no_spatial=False)

        self.cnn_x = nn.Conv1d(in_channels=1 * self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.fm, 1),
                               stride=1,
                               bias=True)

    def forward(self, data):

        if self.mv == 1:
            x_m = data['seq']['data_matrix'].cuda()
            x_m_f_edge_index=data['seq']['edges']


        elif self.mv == 2:
            x_m = data['gip']['data_matrix'].cuda()
            x_m_f_edge_index=data['gip']['edges']

        else:
            raise ValueError("Invalid mv value. It should be 1, 2, or 3.")

        x_m_f_edge_index = torch.tensor(x_m_f_edge_index, dtype=torch.long, device=x_m.device)
        x_m_f1 = torch.relu(self.gat_x1_f(x_m.cuda(), x_m_f_edge_index))  # 第一层
        # x_m_f1 = torch.relu(self.gat_x1_f(x_m, x_m_f_edge_index))  # 第一层
        x_m_f2 = torch.relu(self.gat_x2_f(x_m_f1, x_m_f_edge_index))  # 第二层
        XM = torch.cat((x_m_f1, x_m_f2), 1).t()        #把两层的GAT拼接起来，一层GAT是circRNAs*featrues

        # print("weidu")
        # print(XM.dim(), YD.dim())  # 输出维度
        XM = XM.view(1, 1 * self.args.gcn_layers, self.args.fm, -1)

        # -------------------------- 修改点：替换注意力调用 --------------------------
        # XM_channel_attention = self.ecax(XM)  # 原self.cbamx改为self.ecax
        XM_channel_attention = self.cbamx(XM)  #
        x = self.cnn_x(XM_channel_attention)
        x = x.view(self.args.out_channels, self.args.miRNA_number).t()

        circ_embed = x[:2115, :]  # 前504行为 circRNA 表示
        mirna_embed = x[2115:, :]  # 后420行为 miRNA 表示

        return circ_embed.mm(mirna_embed.t()), circ_embed, mirna_embed