from kan import *
from sympy import *
from scipy.stats import qmc, chatterjeexi
import numpy as np

step_num = 100

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = KAN(width=[2,5,1], grid=10, k=3, device=device)

    def func(x):
        z = x[:,0]/x[:,1] + 3
        return np.exp(z) + np.sin(z) + 5.1
    
    rng = np.random.default_rng()

    sampler = qmc.LatinHypercube(d=2, scramble=True, optimization="lloyd")

    dataset = {
        'train_input': np.array(qmc.scale(sampler.random(n=1000), [1, 1], [2, 2])),
        'test_input': np.array(qmc.scale(sampler.random(n=1000), [1, 1], [2, 2])),
        'train_label': [],
        'test_label': [],
    }

    dataset['train_label'] = np.array([func(dataset['train_input'])]).T
    dataset['test_label'] = np.array([func(dataset['test_input'])]).T

    dataset['train_input'] = torch.tensor(dataset['train_input'])
    dataset['train_label'] = torch.tensor(dataset['train_label'])
    dataset['test_input'] = torch.tensor(dataset['test_input'])
    dataset['test_label'] = torch.tensor(dataset['test_label'])

    def xi_corr_train():
        return (1 - chatterjeexi(model(dataset['train_input']).detach().numpy().T[0], dataset['train_label'].detach().numpy().T[0]).statistic)*100

    def xi_corr_test():
        return (1 - chatterjeexi(model(dataset['test_input']).detach().numpy().T[0], dataset['test_label'].detach().numpy().T[0]).statistic)*100

    model.fit(dataset, opt="LBFGS", steps=step_num, lamb=0.001, metrics=(xi_corr_train, xi_corr_test), loss_fn=torch.nn.CrossEntropyLoss())
    model = model.prune()
    model.fit(dataset, opt="LBFGS", steps=step_num, metrics=(xi_corr_train, xi_corr_test), loss_fn=torch.nn.CrossEntropyLoss())
    model = model.refine(50)
    model.fit(dataset, opt="LBFGS", steps=step_num, metrics=(xi_corr_train, xi_corr_test), loss_fn=torch.nn.CrossEntropyLoss())
    
    model.auto_symbolic()
    model.fit(dataset, opt="LBFGS", steps=step_num, metrics=(xi_corr_train, xi_corr_test), loss_fn=torch.nn.CrossEntropyLoss())

    symbolic_formula = model.symbolic_formula()
    symbolic_formula = sympy.expand(symbolic_formula[0][0])
    symbolic_formula = utils.ex_round(symbolic_formula, 1)
    symbolic_formula = sympy.simplify(symbolic_formula)
    print(symbolic_formula)