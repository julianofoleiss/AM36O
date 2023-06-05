import itertools
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from tqdm.notebook import tqdm

from joblib import delayed, Parallel

from scipy.stats import ttest_ind_from_stats
    
def calcular_estatisticas(resultados):
    return np.mean(resultados), np.std(resultados), np.min(resultados), np.max(resultados)

def imprimir_estatisticas(resultados):
    media, desvio, mini, maxi = calcular_estatisticas(resultados)
    print("Resultados: %.2f +- %.2f, min: %.2f, max: %.2f" % (media, desvio, mini, maxi))
    
def rejeitar_hip_nula(amostra1, amostra2, alpha=0.05):
    media_amostral1, desvio_padrao_amostral1, _, _ = calcular_estatisticas(amostra1)
    media_amostral2, desvio_padrao_amostral2, _, _ = calcular_estatisticas(amostra2)
    
    _, pvalor = ttest_ind_from_stats(media_amostral1, desvio_padrao_amostral1, len(amostra1), media_amostral2, desvio_padrao_amostral2, len(amostra2))
    return (pvalor <= alpha, pvalor)
    

def print_t_tests(resultados, cols=None, alpha=0.05):
    '''
    Esta função imprime o resultado do teste de hipótese nula em uma matriz entre todos os pares de experimentos.
    
    resultados : dict
        Dicionário onde as chaves são strings que representam um experimento e os valores
        são listas com os resultados por fold.
        
    cols : list, optional
        Lista de colunas que serão mostradas. Por padrão, usar todas as chaves do dicionário
        resultados como colunas.
        
    alpha: float, optional
        Limiar para rejeitar a hipótese nula. 
        A hipótese nula será rejeitada se p_valor <= alpha.
    
    Caso a hipótese não possa ser rejeitada (p_valor>alpha), o p_valor é simplesmente impresso. 
    Caso a hipótese seja rejeitada, o p_valor é impresso, justamente com um (*c), onde s é
    um caractere que representa a relação entre as médias da linha e da coluna. Por exemplo, 
    se a média do experimento     da linha for maior que a média do experimento da coluna, 
    c será >. Caso contrário, c será <.
    '''    
    if cols is None:
        cols = sorted(resultados)    
    
    largura = max(max(map(len,cols))+2,12)
    
    print(" " * largura , end="")
    
    for t in cols:
        print(t.center(largura), end='')
    print()
    
    for t in sorted(resultados):
        print(t.center(largura), end='')
        for t2 in cols:
            d, p = rejeitar_hip_nula(resultados[t], resultados[t2], alpha=alpha)
            dif = '<' if np.mean(resultados[t]) - np.mean(resultados[t2]) < 0 else '>'
            print(("%.02f%s" % (p, (' (*%c)' % dif) if d else '')).center(largura), end='')
        print()
    
    
def selecionar_melhor_modelo(classificador, X_treino, X_val, y_treino, y_val, n_jobs=4, 
                             cv_folds=None, params={}):
    
    def treinar_ad(X_treino, X_val, y_treino, y_val, params):
        clf = classificador(**params)
        clf.fit(X_treino, y_treino)
        pred = clf.predict(X_val)
        
        if len(set(y_treino)) > 2:
            return f1_score(y_val, pred, average='weighted')
        else:
            return f1_score(y_val, pred)
    
    
    if cv_folds is not None:
        #Se for pra usar validação cruzada, usar GridSearchCV
        score_fn = 'f1' if len(set(y_treino)) < 3 else 'f1_weighted'
        
        clf = GridSearchCV(classificador(), params, cv=cv_folds, n_jobs=n_jobs, scoring=score_fn)
        #Passar todos os dados (Treino e Validação) para realizar a seleção dos parâmetros.
        clf.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])
        
        melhor_comb = clf.best_params_
        melhor_val = clf.best_score_
        
    else:
        param_grid = list(ParameterGrid(params))
        
        f1s_val = Parallel(n_jobs=n_jobs)(delayed(treinar_ad)
                                         (X_treino, X_val, y_treino, y_val, p) for p in param_grid)

        melhor_val = max(f1s_val)
        melhor_comb = param_grid[np.argmax(f1s_val)]
        
        clf = classificador(**melhor_comb)
        
        clf.fit(np.vstack((X_treino, X_val)), [*y_treino, *y_val])
    
    return clf, melhor_comb, melhor_val

def do_cv(classificador, X, y, cv_splits, param_cv_folds=None, n_jobs=8, scale=False, imputer=None, dim_red=None, params={}):

    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=1)

    f1s = []
    
    pgb = tqdm(total=cv_splits, desc='Folds avaliados')
    
    for treino_idx, teste_idx in skf.split(X, y):

        if isinstance(X, pd.DataFrame):
            X_treino = X.iloc[treino_idx]
            X_teste = X.iloc[teste_idx]
            
            if imputer is not None:
                X_treino = imputer.fit_transform(X_treino)
                X_teste = imputer.transform(X_teste)
            else:
                X_treino = X_treino.values
                X_teste = X_teste.values   
            
        else:
            X_treino = X[treino_idx]
            X_teste = X[teste_idx]
            
        y_treino = y[treino_idx]
        y_teste = y[teste_idx]        

        X_treino, X_val, y_treino, y_val = train_test_split(X_treino, y_treino, stratify=y_treino, test_size=0.2, random_state=1)
        
        if imputer is not None:
            X_val = imputer.transform(pd.DataFrame(X_val, columns=X.columns))
        
        if scale:
            ss = StandardScaler()
            X_treino = ss.fit_transform(X_treino)
            X_teste = ss.transform(X_teste)
            X_val = ss.transform(X_val)
            
        if dim_red is not None:
            X_treino = dim_red.fit_transform(X_treino, y_treino)
            X_teste = dim_red.transform(X_teste)
            X_val = dim_red.transform(X_val)
            
        ad, melhor_comb, _ = selecionar_melhor_modelo(classificador, X_treino, X_val, y_treino, y_val, 
                                                      n_jobs=n_jobs, cv_folds=param_cv_folds, params=params)
        pred = ad.predict(X_teste)

        if len(set(y_treino)) > 2:
            f1 = f1_score(y_teste, pred, average='weighted')
        else:
            f1 = f1_score(y_teste, pred)
        f1s.append(f1)
        
        pgb.update(1)
        
    pgb.close()
    
    return f1s