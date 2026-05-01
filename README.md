# Programming-for-AI-assignment1
task

```bash
pip install -e .
python -m pytest -q

git link shared : https://github.com/Gavin872/Programming-for-AI-assignment1

python -c "import numpy as np; from numcompute.preprocessing import Imputer, StandardScaler; from numcompute.pipeline import Pipeline; X=np.array([[1.0,np.nan],[2.0,5.0],[3.0,6.0]]); pipe=Pipeline([('imputer',Imputer(fill_value=0.0)),('scaler',StandardScaler())]); print('Original data:'); print(X); print('Pipeline output:'); print(pipe.fit_transform(X))"
python -c "import numpy as np; from numcompute.metrics import accuracy, precision, recall, f1, mse; y_true=np.array([1,0,1,1]); y_pred=np.array([1,0,0,1]); print('Accuracy:', accuracy(y_true,y_pred)); print('Precision:', precision(y_true,y_pred)); print('Recall:', recall(y_true,y_pred)); print('F1:', f1(y_true,y_pred)); print('MSE:', mse(np.array([1.,2.,3.]), np.array([1.,2.,4.])))"
python -c "import numpy as np; from numcompute.rank import rank, percentile; scores=np.array([10,20,20,30]); values=np.array([10,4,8,2,9]); print('Average rank:', rank(scores, method='average')); print('Dense rank:', rank(scores, method='dense')); print('50th percentile:', percentile(values, 50))"
python -c "import numpy as np; from numcompute.optim import grad; f=lambda v: v[0]**2+v[1]**2; print('Gradient:', grad(f, np.array([3.0,4.0])))"











