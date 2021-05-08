

#I) Intégration

# #### Importations des librairies

import warnings
warnings.filterwarnings('ignore')

# In[2]:

# Format des données
import pandas as pd
import numpy as np

# Outils de graphs
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Scaling
from sklearn.preprocessing import scale

# Temps
import time

# Machine Learning Models - Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
    
# Evaluation
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# Amélioration des modèles 
from sklearn.model_selection import GridSearchCV

# #### Importation des données

# In[3]:

X_train = pd.read_csv('./Données projet Qube RT/x_train_Lafd4AH.csv', sep = ',')
X_test = pd.read_csv('./Données projet Qube RT/x_test_c7ETL4q.csv', sep = ',')
y_train = pd.read_csv('./Données projet Qube RT/y_train_JQU4vbI.csv', sep = ',')
y_test = pd.read_csv('./Données projet Qube RT/test_rand.csv', sep = ',')


# In[4]:

X_train.head()

# In[5]:

y_train.head()

# # II) Data Cleaning

# #### Y a-t-il des values non complétées (NaN) ?

# In[6]:

X_train.shape

# In[7]:

X_train.info()


# In[8]:

X_train = X_train.dropna()

# In[9]:

X_train.reset_index(inplace = True, drop = True)

# In[10]:

X_train.shape

# # III) Data Exploration

# ## III.a) Analyses des distributions

# #### Distribution des données suivant les Labels

# In[11]:

plt.figure(figsize=(7,5))
sns.countplot(x='RET', data=y_test)
plt.ylabel('Nombre')
plt.xlabel('Valeur RET')
plt.title("Distribution des labels")
plt.show()

# #### Distribution des industries

# In[12]:

data_train = X_train.merge(y_train,
                          how = 'left',
                          on = 'ID',
                          validate = '1:1')

# In[13]:

plt.figure(figsize=(16,5))
sns.countplot(x='INDUSTRY', hue = 'RET', data=data_train)
plt.ylabel("Nombre d'apparitions")
plt.xlabel('Industries')
plt.title("Distribution des labels par Industrie")
plt.show()


# In[14]:

plt.figure(figsize=(16,5))
sns.countplot(x='INDUSTRY_GROUP', hue = 'RET', data=data_train)
plt.ylabel("Nombre d'apparitions")
plt.xlabel('Industries')
plt.title("Distribution des labels par Industrie Group")
plt.show()

# #### Répartition des secteurs

# In[15]:

plt.figure(figsize=(16,5))
sns.countplot(x='SECTOR', hue = 'RET', data=data_train)
plt.ylabel("Nombre d'apparitions")
plt.xlabel('Industries')
plt.title("Distribution des labels par Secteur")
plt.show()

# #### Répartition des sous industries

# In[16]:

plt.figure(figsize=(16,5))
sns.countplot(x='SUB_INDUSTRY', hue = 'RET', data=data_train)
plt.ylabel("Nombre d'apparitions")
plt.xlabel('Industries')
plt.title("Distribution des labels par Sous Industrie")
plt.show()

# ## III.b) Analyses des corrélations

# #### Correlation entre les volumes

# In[17]:

columns = ['VOLUME_' + str(i) for i in range(1,21)]

# In[18]:

corr = data_train[columns].corr()
ax, fig = plt.subplots(figsize=(15,15))
sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True)
plt.show()

# #### Correlation entre les Retours

# In[19]:

columns = ['RET_' + str(i) for i in range(1,21)]

# In[20]:

corr = data_train[columns].corr()
ax, fig = plt.subplots(figsize=(15,15))
sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True)
plt.show()

# #### Correlation entre les Retours et Volumes

# In[21]:

columns = ['RET_' + str(i) for i in range(1,5)]
columns += ['VOLUME_' + str(i) for i in range(1,5)]

# In[22]:

corr = data_train[columns].corr()
ax, fig = plt.subplots(figsize=(15,15))
sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True)
plt.show()

# # IV) Features Engineering

# ## IV.1) Créations de nouvelles features

# ### IV.1.A) Création de nouvelles features - première couche

# In[23]:

data_train = X_train.merge(y_train,
                          how = 'left',
                          on = 'ID',
                          validate = '1:1')

# In[24]:

### Creation de nouvelles features

# Moyenne de RET_1 conditionnellement à la Date et au Secteur
means_ret1_by_date_sector = data_train.groupby(['DATE','SECTOR']).agg({'RET_1' : 'mean'}).reset_index()
means_ret1_by_date_sector.columns = ['DATE','SECTOR','Mean_RET1_by_Date_Sector']

# Moyenne de RET_1 conditionnellement à la Date et à l'Industrie
means_ret1_by_date_industry = data_train.groupby(['DATE','INDUSTRY']).agg({'RET_1' : 'mean'}).reset_index()
means_ret1_by_date_industry.columns = ['DATE','INDUSTRY','Mean_RET1_by_Date_Industry']

# Moyenne de Volume_1 conditionnellement à la Date et au Secteur
means_Vol1_by_date_sector = data_train.groupby(['DATE','SECTOR']).agg({'VOLUME_1' : 'mean'}).reset_index()
means_Vol1_by_date_sector.columns = ['DATE','SECTOR','Mean_Vol1_by_Date_Sector']

# Moyenne de Volume_1 conditionnellement à la Date et à l'Industrie
means_Vol1_by_date_industry = data_train.groupby(['DATE','INDUSTRY']).agg({'VOLUME_1' : 'mean'}).reset_index()
means_Vol1_by_date_industry.columns = ['DATE','INDUSTRY','Mean_Vol1_by_Date_Industry']


# In[25]:

### Ajout des colonnes aux données

data_train = data_train.merge(means_ret1_by_date_sector,
                             how = 'left',
                             on = ['DATE','SECTOR'],
                             validate = 'm:1')

data_train = data_train.merge(means_ret1_by_date_industry,
                             how = 'left',
                             on = ['DATE','INDUSTRY'],
                             validate = 'm:1')

data_train = data_train.merge(means_Vol1_by_date_sector,
                             how = 'left',
                             on = ['DATE','SECTOR'],
                             validate = 'm:1')

data_train = data_train.merge(means_Vol1_by_date_industry,
                             how = 'left',
                             on = ['DATE','INDUSTRY'],
                             validate = 'm:1')


# #### Correlation entre RET et les nouvelles features - première couche

# In[26]:

columns = ['RET',
       'Mean_RET1_by_Date_Sector', 'Mean_RET1_by_Date_Industry',
       'Mean_Vol1_by_Date_Sector', 'Mean_Vol1_by_Date_Industry']


# In[27]:

corr = data_train[columns].corr()
ax, fig = plt.subplots(figsize=(8,8))
sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True)
plt.show()

# ### IV.1.B) Création de nouvelles features - deuxième couche

# In[28]:

### Creation de nouvelles features

# Moyenne de RET_1 conditionnellement à la Date et au groupe d'industrie
means_ret1_by_date_GrpInd = data_train.groupby(['DATE','INDUSTRY_GROUP']).agg({'RET_1' : 'mean'}).reset_index()
means_ret1_by_date_GrpInd.columns = ['DATE','INDUSTRY_GROUP','Mean_RET1_by_Date_GrpInd']

# Moyenne de RET_1 conditionnellement à la Date et à la sous industrie
means_ret1_by_date_SubInd = data_train.groupby(['DATE','SUB_INDUSTRY']).agg({'RET_1' : 'mean'}).reset_index()
means_ret1_by_date_SubInd.columns = ['DATE','SUB_INDUSTRY','Mean_RET1_by_Date_SubInd']

# Moyenne de Volume_1 conditionnellement à la Date et au groupe d'industrie
means_Vol1_by_date_GrpInd = data_train.groupby(['DATE','INDUSTRY_GROUP']).agg({'VOLUME_1' : 'mean'}).reset_index()
means_Vol1_by_date_GrpInd.columns = ['DATE','INDUSTRY_GROUP','Mean_Vol1_by_Date_GrpInd']

# Moyenne de Volume_1 conditionnellement à la Date et à la sous industrie
means_Vol1_by_date_SubInd = data_train.groupby(['DATE','SUB_INDUSTRY']).agg({'VOLUME_1' : 'mean'}).reset_index()
means_Vol1_by_date_SubInd.columns = ['DATE','SUB_INDUSTRY','Mean_Vol1_by_Date_SubInd']


# In[29]:

### Ajout des colonnes aux données

data_train = data_train.merge(means_ret1_by_date_GrpInd,
                             how = 'left',
                             on = ['DATE','INDUSTRY_GROUP'],
                             validate = 'm:1')

data_train = data_train.merge(means_ret1_by_date_SubInd,
                             how = 'left',
                             on = ['DATE','SUB_INDUSTRY'],
                             validate = 'm:1')

data_train = data_train.merge(means_Vol1_by_date_GrpInd,
                             how = 'left',
                             on = ['DATE','INDUSTRY_GROUP'],
                             validate = 'm:1')

data_train = data_train.merge(means_Vol1_by_date_SubInd,
                             how = 'left',
                             on = ['DATE','SUB_INDUSTRY'],
                             validate = 'm:1')


# #### Correlation entre RET et les nouvelles features - seconde couche

# In[30]:


columns = ['RET',
       'Mean_RET1_by_Date_GrpInd', 'Mean_RET1_by_Date_SubInd',
       'Mean_Vol1_by_Date_GrpInd', 'Mean_Vol1_by_Date_SubInd']


# In[31]:


corr = data_train[columns].corr()
ax, fig = plt.subplots(figsize=(8,8))
sns.heatmap(corr, vmin=-1, cmap='coolwarm', annot=True)
plt.show()

# ## IV.2) Préparation du Dataset

# In[32]:


df_train = data_train[['RET_1', 'VOLUME_1', 'RET_2', 'VOLUME_2', 'RET_3',
                       'VOLUME_3', 'RET_4', 'VOLUME_4', 'RET_5', 'VOLUME_5', 'RET',
                       'Mean_RET1_by_Date_Sector', 'Mean_Vol1_by_Date_Sector',
                       'Mean_RET1_by_Date_SubInd']]


# In[33]:


df_train.head()


# # V) Classification, Test et Optimisation

# ## V.I) Modèle général

# ### V.I.1) Construction du modèle

# #### Choix du modèle

# In[34]:


models = []
models.append(('Logistic Regression',LogisticRegression()))
models.append(('Decision Tree Classifier',DecisionTreeClassifier()))
models.append(('Random Forest Classifier',RandomForestClassifier()))


# ##### Normalization

# In[35]:


X1 = df_train.loc[:,['RET_1', 'VOLUME_1', 'RET_2', 'VOLUME_2', 'RET_3',
                   'VOLUME_3', 'RET_4', 'VOLUME_4', 'RET_5', 'VOLUME_5',
                   'Mean_RET1_by_Date_Sector', 'Mean_Vol1_by_Date_Sector',
                   'Mean_RET1_by_Date_SubInd']]
y1 = df_train.RET
X1_scale = (X1-X1.max())/(X1.max() - X1.min())

# ##### Cross Validation

# In[36]:

accuracy_results = list()

index = [m[0] for m in models]
for nom_model,model in models:
    print(nom_model)
    cv_results = cross_val_score(model, X1_scale, y1, cv=7, scoring='accuracy')
    accuracy_results.append(cv_results.mean()*100)
    
Performance_results_scaled = pd.DataFrame(accuracy_results , index=index , columns= ['Performance en %'])


# In[37]:


Performance_results_scaled

# In[38]:


best_clf = RandomForestClassifier()
best_clf.get_params


# In[39]:

param_grid={'max_depth' : [8,12], 'n_estimators' : [100,200]}


# In[40]:

clf = GridSearchCV(best_clf,param_grid, cv=3, scoring = 'accuracy')


# In[41]:

tic = time.time()
clf.fit(X1_scale,y1)
toc = time.time()
print(toc-tic)


# In[42]:


clf.best_score_


# In[43]:


clf.best_params_


# ## V.II) Modèle par Industrie 

# #### Choix du modèle par Industrie

# In[44]:

def get_data_per_industrie(ind):
    data = data_train[data_train.INDUSTRY == ind]
    X = data.loc[:,['RET_1', 'VOLUME_1', 'RET_2', 'VOLUME_2', 'RET_3',
                   'VOLUME_3', 'RET_4', 'VOLUME_4', 'RET_5', 'VOLUME_5',
                   'Mean_RET1_by_Date_Sector', 'Mean_Vol1_by_Date_Sector',
                   'Mean_RET1_by_Date_SubInd']]
    
    X = (X-X.max())/(X.max() - X.min())
    y = data.RET
    return X, y


# In[45]:


models = []
models.append(('Logistic Regression',LogisticRegression()))
models.append(('Decision Tree Classifier',DecisionTreeClassifier()))
models.append(('Random Forest Classifier',RandomForestClassifier()))


# In[46]:


accuracy_results = []
ind_list = []
mod = []
index = [m[0] for m in models]

for ind in data_train.INDUSTRY.unique():
    for nom_model,model in models:
        print(ind, nom_model)
        X, y = get_data_per_industrie(ind)
        cv_results = cross_val_score(model, X, y, cv=7, scoring='accuracy')
        accuracy_results.append(cv_results.mean()*100)
        mod.append(nom_model)
        ind_list.append(ind)


# In[47]:


Performance_results_ind = pd.DataFrame( columns= ['Industrie','Model','Performance en %'])
Performance_results_ind['Industrie'] = ind_list
Performance_results_ind['Model'] = mod
Performance_results_ind['Performance en %'] = accuracy_results


# #### On garde le meilleur modèle pour chaque industrie

# In[48]:


A = Performance_results_ind.groupby(['Industrie']).agg({'Performance en %':'max'}).reset_index()
Best_model = A.merge(Performance_results_ind,
           how ='left',
           on = ['Industrie','Performance en %'])


# In[49]:


Best_model.head()


# #### Calcul de la moyenne

# In[50]:


Best_model['Performance en %'].mean()

# # VI) Prédiction finale

# ## VI.1) Mise en forme du dataset de Test

# #### Retrait des Nan

# In[51]:


# Retrait des Nan
X_test.dropna(inplace = True)


# #### Ajout des nouvelles features

# In[52]:


### Creation de nouvelles features

# Moyenne de RET_1 conditionnellement à la Date et au Secteur
means_ret1_by_date_sector = X_test.groupby(['DATE','SECTOR']).agg({'RET_1' : 'mean'}).reset_index()
means_ret1_by_date_sector.columns = ['DATE','SECTOR','Mean_RET1_by_Date_Sector']

# Moyenne de Volume_1 conditionnellement à la Date et au Secteur
means_Vol1_by_date_sector = X_test.groupby(['DATE','SECTOR']).agg({'VOLUME_1' : 'mean'}).reset_index()
means_Vol1_by_date_sector.columns = ['DATE','SECTOR','Mean_Vol1_by_Date_Sector']

# Moyenne de RET_1 conditionnellement à la Date et à la sous industrie
means_ret1_by_date_SubInd = X_test.groupby(['DATE','SUB_INDUSTRY']).agg({'RET_1' : 'mean'}).reset_index()
means_ret1_by_date_SubInd.columns = ['DATE','SUB_INDUSTRY','Mean_RET1_by_Date_SubInd']


# In[53]:


### Ajout des colonnes aux données

X_test = X_test.merge(means_ret1_by_date_sector,
                             how = 'left',
                             on = ['DATE','SECTOR'],
                             validate = 'm:1')

X_test = X_test.merge(means_Vol1_by_date_sector,
                             how = 'left',
                             on = ['DATE','SECTOR'],
                             validate = 'm:1')

X_test = X_test.merge(means_ret1_by_date_SubInd,
                             how = 'left',
                             on = ['DATE','SUB_INDUSTRY'],
                             validate = 'm:1')


# #### Sélection des variables

# In[54]:


df_test = X_test.merge(y_test,
                      how = 'left',
                      on = 'ID')


# In[55]:


#  On normalise les données : 

X1_test = df_test.loc[:,['RET_1', 'VOLUME_1', 'RET_2', 'VOLUME_2', 'RET_3',
                   'VOLUME_3', 'RET_4', 'VOLUME_4', 'RET_5', 'VOLUME_5',
                   'Mean_RET1_by_Date_Sector', 'Mean_Vol1_by_Date_Sector',
                   'Mean_RET1_by_Date_SubInd']]
y1_test = df_test.RET
X1_test_scale = scale(X1_test)


# ## VI.2) Modèle général

# ### VI.2.A) Entrainement du modèle général

# In[56]:


model_général = RandomForestClassifier(max_depth = 12, n_estimators = 200)


# In[57]:


model_général.fit(X1_scale,y1)


# ### VI.2.B) Prédiction du Test Set avec le modèle général

# #### Prédiction

# In[58]:


y_pred = model_général.predict(X1_test_scale)


# #### Evaluation

# In[59]:


accuracy_score(y1_test,y_pred)

# ## VI.3) Modèle par industrie

# ### VI.3.A) Entrainement des modèles

# #### Ajout des modèles à la base des meilleurs modèles

# In[60]:


models = { 'Logistic Regression' : LogisticRegression(),
            'Decision Tree Classifier' : DecisionTreeClassifier(),
            'Random Forest Classifier' : RandomForestClassifier()}


# In[61]:


Best_model['Model_code'] = Best_model.Model.map(models)


# In[62]:


Best_model.head()


# #### Entrainement pour chaque modèle

# In[63]:


for i in Best_model.index:
    ind = Best_model.loc[i,'Industrie']
    print("Industrie en cours d'entrainement : " , ind)
    mod = Best_model.loc[i,'Model_code']
    X, y = get_data_per_industrie(ind)
    mod.fit(X,y)


# ### VI.3.B) Prédiction du Test Set avec le modèle par Industrie

# Certaines industries présentes dans le jeu de test ne sont pas dans le jeu d'entrainement. Dans ces cas là, nous lançons une prédiction avec le modèle général.

# #### Prédiction par Industrie

# In[86]:


def get_data_per_industrie_test(ind):
    data = df_test[df_test.INDUSTRY == ind]
    X = data.loc[:,['RET_1', 'VOLUME_1', 'RET_2', 'VOLUME_2', 'RET_3',
                   'VOLUME_3', 'RET_4', 'VOLUME_4', 'RET_5', 'VOLUME_5',
                   'Mean_RET1_by_Date_Sector', 'Mean_Vol1_by_Date_Sector',
                   'Mean_RET1_by_Date_SubInd']]
    
    X = (X-X.max())/(X.max() - X.min())
    y = data.RET
    return X, y


# In[65]:


list_industrie = []
valeur_accuracy = []
for ind in df_test.INDUSTRY.unique():
    print("Prédiction de l'industrie : ",ind)
    list_industrie.append(ind)
    try: 
        X,y = get_data_per_industrie_test(ind)
        model = Best_model.loc[Best_model.Industrie == ind, 'Model_code'].values[0]
        y_pred = model.predict(X)
        valeur_accuracy.append(accuracy_score(y,y_pred))
        
    except:
        X,y = get_data_per_industrie_test(ind)
        y_pred = model_général.predict(X)
        valeur_accuracy.append(accuracy_score(y,y_pred))


# In[66]:


prediction_per_industrie = pd.DataFrame()
prediction_per_industrie['Industrie'] = list_industrie
prediction_per_industrie['Valeur_accuracy'] = valeur_accuracy


# In[67]:


prediction_per_industrie.head()


# #### Evaluation

# In[68]:


prediction_per_industrie.Valeur_accuracy.mean()

# ## VI.4) Modèle mixte : général et par industrie

# In[69]:


Best_model_Mixte = Best_model.copy()

# In[70]:


Best_model_Mixte.loc[Best_model_Mixte['Performance en %'] < 50.2,'Model'] = 'Model General'


# In[71]:


Best_model_Mixte.loc[Best_model_Mixte.Model == 'Model General','Model_code'] = 'model_général'


# In[72]:


Best_model_Mixte.head()


# ### VI.4.A) Entrainement des modèles

# ### VI.4.B) Prédiction du Test Set avec le modèle mixte

# #### Prédiction par Industrie

# In[73]:


def get_data_per_industrie_test(ind):
    data = df_test[df_test.INDUSTRY == ind]
    X = data.loc[:,['RET_1', 'VOLUME_1', 'RET_2', 'VOLUME_2', 'RET_3',
                   'VOLUME_3', 'RET_4', 'VOLUME_4', 'RET_5', 'VOLUME_5',
                   'Mean_RET1_by_Date_Sector', 'Mean_Vol1_by_Date_Sector',
                   'Mean_RET1_by_Date_SubInd']]
    
    X = (X-X.max())/(X.max() - X.min())
    y = data.RET
    return X, y


# In[78]:


list_industrie = []
valeur_accuracy = []
for ind in df_test.INDUSTRY.unique():
    print("Prédiction de l'industrie : ",ind)
    list_industrie.append(ind)
    X,y = get_data_per_industrie_test(ind)
    
    try: 
        name_model = Best_model_Mixte.loc[Best_model_Mixte.Industrie == ind, 'Model'].values[0]
        
        if name_model != 'Model General':
       
            model = Best_model_Mixte.loc[Best_model_Mixte.Industrie == ind, 'Model_code'].values[0]
            y_pred = model.predict(X)
            valeur_accuracy.append(accuracy_score(y,y_pred))

        else:
            y_pred = model_général.predict(X)
            valeur_accuracy.append(accuracy_score(y,y_pred))
    except:
        y_pred = model_général.predict(X)
        valeur_accuracy.append(accuracy_score(y,y_pred))


# In[79]:


prediction_mixte = pd.DataFrame()
prediction_mixte['Industrie'] = list_industrie
prediction_mixte['Valeur_accuracy'] = valeur_accuracy


# In[80]:


prediction_mixte


# In[81]:


prediction_mixte.Valeur_accuracy.mean()


# ## VI.5) Conclusions





