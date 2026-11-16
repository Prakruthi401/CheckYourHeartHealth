"""PTB-XL signal classifier. Research use only; never accepts ECG images."""
from pathlib import Path
import io, base64, zlib
import numpy as np
import torch
torch.set_num_threads(1)
from torch import nn
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
class Blk(nn.Module):
    def __init__(s,i,o,st):
        super().__init__(); s.c1=nn.Conv1d(i,o,7,st,3,bias=False); s.b1=nn.BatchNorm1d(o); s.c2=nn.Conv1d(o,o,7,1,3,bias=False); s.b2=nn.BatchNorm1d(o)
        s.sk=nn.Sequential(nn.Conv1d(i,o,1,st,bias=False),nn.BatchNorm1d(o))
    def forward(s,x):
        y=torch.relu(s.b1(s.c1(x))); y=s.b2(s.c2(y)); return torch.relu(y+s.sk(x))
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.f=nn.Sequential(nn.Conv1d(12,32,15,2,7,bias=False),nn.BatchNorm1d(32),nn.ReLU(),Blk(32,32,1),Blk(32,64,2),Blk(64,96,2),Blk(96,128,2),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Dropout(0.3),nn.Linear(128,5))
    def forward(s,x): return s.f(x)
NORMALIZATION = np.array([[-0.0005988287157379091, -0.00018327069119550288, 0.00041516561759635806, 0.0003774944634642452, -0.0004609675379469991, 0.00013296003453433514, 0.00018647403339855373, -0.0006973108975216746, -0.0013283761218190193, -0.0010195443173870444, -0.0005862615071237087, -0.0005335180321708322], [0.1550021916627884, 0.16399142146110535, 0.15535008907318115, 0.13945253193378448, 0.13170994818210602, 0.13954824209213257, 0.2312672883272171, 0.3295155167579651, 0.3271399736404419, 0.29607445001602173, 0.2787685692310333, 0.2616052031517029]], dtype=np.float32)

class ECGClassifier:
    def __init__(self):
        self.model = None
    def load_models(self):
        if self.model is None:
            folder = Path(__file__).resolve().parent
            encoded = ''.join((folder / f'ecg_weights_{i}.b64').read_text().strip() for i in range(5))
            state = torch.load(io.BytesIO(zlib.decompress(base64.b64decode(encoded))), map_location='cpu', weights_only=True)
            model = Net()
            model.load_state_dict(state)
            model.eval()
            self.model = model
        return self.model
    def predict(self, signal):
        signal = np.asarray(signal, dtype=np.float32)
        if signal.shape != (1000, 12) or not np.isfinite(signal).all():
            raise ValueError('Expected 1000 rows x 12 leads, all finite, in millivolts at 100 Hz.')
        if np.max(np.abs(signal)) > 30:
            raise ValueError('Values exceed 30 mV. Check signal units; raw ADC counts are not supported.')
        x = torch.from_numpy((signal-NORMALIZATION[0])/NORMALIZATION[1]).T.unsqueeze(0)
        with torch.no_grad():
            scores = torch.sigmoid(self.load_models()(x))[0].numpy()
        return dict(zip(CLASSES, map(float, scores)))
ecg_classifier = ECGClassifier()
