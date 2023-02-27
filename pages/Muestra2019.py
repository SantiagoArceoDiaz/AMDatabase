#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd #importa la paquetería PAnel DAta (llamada pandas)
import matplotlib
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted # importa paqueteria para graficar diagramas de venn
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt #importa pyplot para hacer gáficas
from matplotlib import numpy as np #importar numpy
import seaborn as sn
import altair as alt
#!pip install altair_catplot
import altair_catplot as altcat
#!pip install xlsxwriter
import xlsxwriter
import os
#!pip install pingouin
get_ipython().run_line_magic('pprint', '')


# In[2]:


#carga los datos de los archivos de excel con los resultados del test de Barthel
df2019 = pd.read_excel('2019BARTH.xlsx')
df2019 = df2019.dropna() #quita las filas que tengan NaN en algun valor
df2019


# In[3]:


Listadf2019=df2019['Nombre'].tolist() # crea una lista con los usuarios de 2019
Setdf2019=set(Listadf2019) # crea un conjunto a partir de la lista de usuarios de 2019


# In[4]:


import pingouin as pg

pg.cronbach_alpha(data=df2019[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete',
       'B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina']])


# In[6]:


df2019M=df2019.loc[df2019['Sexo']==1]
df2019M.head(121)


# In[7]:


import pingouin as pg

pg.cronbach_alpha(data=df2019M[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete',
       'B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina']])

#(0.7734375, array([0.336, 0.939]))


# In[8]:


hist = df2019M['Edad'].hist(bins=10)
plt.xlabel("Edad (años)")
 
# Label for y-axis
plt.ylabel("Número de individuos")
plt.savefig("Mujeres2019.png", bbox_inches='tight', dpi=300)


# In[9]:


M2019E60=df2019M.loc[(df2019M['Edad']<=60)]
M2019E60


# In[10]:


M2019E60.describe()


# In[11]:


M2019E70=df2019M.loc[((df2019M['Edad']<=70) & (df2019['Edad']>60))]
M2019E70


# In[12]:


M2019E70.describe()


# In[13]:


M2019E80=df2019M.loc[((df2019M['Edad']<=80) & (df2019['Edad']>70))]
M2019E80


# In[14]:


M2019E80.describe()


# In[15]:


M2019E80p=df2019M.loc[(df2019['Edad']>80)]
M2019E80p


# In[16]:


M2019E80p.describe()


# In[17]:


df2019H=df2019.loc[df2019['Sexo']==2]
df2019H


# In[18]:


import pingouin as pg

pg.cronbach_alpha(data=df2019H[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete',
       'B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina']])


# In[19]:


hist = df2019H['Edad'].hist(bins=7)  

plt.xlabel("Edad (años)")
 
# Label for y-axis
plt.ylabel("Número de individuos")
plt.savefig("Hombres2019.png", bbox_inches='tight', dpi=300)

#plt.hist(df['Age'], bins=[20,25,35,40,45,50])


# In[20]:


H2019E60=df2019H.loc[(df2019H['Edad']<=60)]
H2019E60


# In[21]:


H2019E70=df2019H.loc[((df2019H['Edad']<=70) & (df2019['Edad']>60))]
H2019E70


# In[22]:


H2019E70.describe()


# In[24]:


H2019E80=df2019H.loc[((df2019H['Edad']<=80) & (df2019['Edad']>70))]
H2019E80


# In[25]:


H2019E80.describe()


# In[26]:


H2019E80p=df2019H.loc[(df2019['Edad']>80)]
H2019E80p


# In[27]:


H2019E80p.describe()


# In[29]:


df2019BS=df2019[['Nombre','Sexo','Edad','B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete',
       'B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina',
       'Int_Barthel']]


# In[30]:


df2019BS


# In[31]:


from operator import index
Xindep=df2019BS.loc[df2019BS['Int_Barthel']==0.0]
Xindepset=set(df2019BS.loc[df2019BS['Int_Barthel']==0.0].index)
Xindepset


# In[32]:


from operator import index
Xdepl=df2019BS.loc[df2019BS['Int_Barthel']==1.0]
Xdeplset=set(df2019BS.loc[df2019BS['Int_Barthel']==1.0].index)
Xdeplset


# In[33]:


from operator import index
Xdepm=df2019BS.loc[df2019BS['Int_Barthel']==2.0]
Xdepmset=set(df2019BS.loc[df2019BS['Int_Barthel']==2.0].index)
Xdepmset


# In[34]:


from operator import index
Xdeps=df2019BS.loc[df2019BS['Int_Barthel']==3.0]
Xdepsset=set(df2019BS.loc[df2019BS['Int_Barthel']==3.0].index)
Xdepsset


# In[35]:


from operator import index
Xdept=df2019BS.loc[df2019BS['Int_Barthel']==4.0]
Xdeptset=set(df2019BS.loc[df2019BS['Int_Barthel']==4.0].index)
Xdeptset


# In[36]:


Xindep.hist(figsize=[14, 12])
plt.savefig("Xindep2019.png", bbox_inches='tight', dpi=300)


# In[37]:


Xdepl.hist(figsize=[14, 12])
plt.savefig("Xdepl2019.png", bbox_inches='tight', dpi=300)


# In[38]:


Xdepm.hist(figsize=[14, 12])
plt.savefig("Xdepm2019.png", bbox_inches='tight', dpi=300)


# In[39]:


Xdeps.hist(figsize=[14, 12])
plt.savefig("Xdeps2019.png", bbox_inches='tight', dpi=300)


# In[41]:


df2019BS.columns[3:-1]


# In[42]:


#a program to find the lower approximation of a feature/ set of features
#mohana palaka
#2019

import pandas as pd
import numpy as np
import time

def indiscernibility(attr, table):

	u_ind = {}	#an empty dictionary to store the elements of the indiscernibility relation (U/IND({set of attributes}))
	attr_values = []	#an empty list to tore the values of the attributes

	for i in (table.index):

		attr_values = []
		for j in (attr):

			attr_values.append(table.loc[i, j])	#find the value of the table at the corresponding row and the desired attribute and add it to the attr_values list

		#convert the list to a string and check if it is already a key value in the dictionary
		key = ''.join(str(k) for k in (attr_values))

		if(key in u_ind):	#if the key already exists in the dictionary
			u_ind[key].add(i)

		else:	#if the key does not exist in the dictionary yet
			u_ind[key] = set()
			u_ind[key].add(i)

	return list(u_ind.values())



def lower_approximation(R, X):	#We have to try to describe the knowledge in X with respect to the knowledge in R; both are LISTS OS SETS [{},{}]

	l_approx = set()	#change to [] if you want the result to be a list of sets

	#print("X : " + str(len(X)))
	#print("R : " + str(len(R)))

	for i in range(len(X)):
		for j in range(len(R)):

			if(R[j].issubset(X[i])):
				l_approx.update(R[j])	#change to .append() if you want the result to be a list of sets

	return l_approx



def gamma_measure(describing_attributes, attributes_to_be_described, U, table):	#a functuon that takes attributes/features R, X, and the universe of objects

	f_ind = indiscernibility(describing_attributes, table)
	t_ind = indiscernibility(attributes_to_be_described, table)

	f_lapprox = lower_approximation(f_ind, t_ind)

	return len(f_lapprox)/len(U)

	#return mod_l_approx(l_approx)/len(U)


def quick_reduct(C, D, table):	#C is the set of all conditional attributes; D is the set of decision attributes

	reduct = set()

	gamma_C = gamma_measure(C, D, table.index, table)
	print(gamma_C)
	gamma_R = 0

	while(gamma_R < gamma_C):

		T = reduct

		for x in (set(C) - reduct):

			feature = set()	#creating a new set to hold the currently selected feature
			feature.add(x)

			print(feature)

			new_red = reduct.union(feature)	#directly unioning x separates the alphabets of the feature...

			gamma_new_red = gamma_measure(new_red, D, table.index, table)
			gamma_T = gamma_measure(T, D, table.index, table)

			if(gamma_new_red > gamma_T):

				T = reduct.union(feature)
				print("added")

		reduct = T

		#finding the new gamma measure of the reduct

		gamma_R = gamma_measure(reduct, D, table.index, table)
		print(gamma_R)

	return reduct


t1 = time.time()

#final_reduct = quick_reduct(audio.columns[0:-1], [audio.columns[-1]], audio)
#final_reduct = quick_reduct(mushroom.columns[1:], [mushroom.columns[0]], mushroom)

final_reduct=quick_reduct(df2019BS.columns[3:-1],[df2019BS.columns[-1]],df2019BS)
print("Serial took : ", str(time.time() - t1))
print(final_reduct)
'''
w_ind = indiscernibility(['weak'], my_table)
d_ind = indiscernibility(['flu'], my_table)
w_gamma = gamma_measure(['weak'], ['flu'], my_table.index)
print(w_gamma)
'''     


# In[275]:


{'B.Retrete', 'B.Heces', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Silla', 'B.Orina', 'B.Comer'}


def low_approximation(R, X):	#We have to try to describe the knowledge in X with respect to the knowledge in R; both are LISTS OS SETS [{},{}]

	low_approx = set()	#change to [] if you want the result to be a list of sets

	#print("X : " + str(len(X)))
	#print("R : " + str(len(R)))

	for j in range(len(R)):

			if(R[j].issubset(X)):
				low_approx.update(R[j])	#change to .append() if you want the result to be a list of sets

	return low_approx


# In[276]:


def upper_approximation(R, X):	#We have to try to describe the knowledge in X with respect to the knowledge in R; both are LISTS OS SETS [{},{}]

	u_approx = set()	#change to [] if you want the result to be a list of sets

	#print("X : " + str(len(X)))
	#print("R : " + str(len(R)))

	for j in range(len(R)):

			if(R[j].intersection(X)!=set()):
				u_approx.update(R[j])	#change to .append() if you want the result to be a list of sets

	return u_approx


# In[307]:


U=df2019BS.columns[1:-1]
U


# In[311]:


Red2019BS=df2019[['Nombre','Sexo','Edad','B.Vestirse', 'B.Desplaz', 'B.Orina', 'B.Silla', 'B.Escal', 'B.Retrete', 'B.Comer','Int_Barthel']]
Red2019BS


# In[312]:


from operator import index
XRindep=Red2019BS.loc[Red2019BS['Int_Barthel']==0.0]
XRindepset=set(Red2019BS.loc[Red2019BS['Int_Barthel']==0.0].index)
XRindepset


# In[387]:


len(XRindepset)


# In[292]:


XRindep[['B.Vestirse', 'B.Desplaz', 'B.Orina', 'B.Silla', 'B.Escal', 'B.Retrete', 'B.Comer','Int_Barthel']].hist(figsize=[14, 12])

plt.savefig("XRindep2019.png", bbox_inches='tight', dpi=300)


# In[287]:


from operator import index
XRdepl=Red2019BS.loc[Red2019BS['Int_Barthel']==1.0]
XRdeplset=set(Red2019BS.loc[Red2019BS['Int_Barthel']==1.0].index)
XRdeplset


# In[388]:


len(XRdeplset)


# In[293]:


XRdepl[['B.Vestirse', 'B.Desplaz', 'B.Orina', 'B.Silla', 'B.Escal', 'B.Retrete', 'B.Comer','Int_Barthel']].hist(figsize=[14, 12])

plt.savefig("XRdep2019l.png", bbox_inches='tight', dpi=300)


# In[290]:


from operator import index
XRdepm=Red2019BS.loc[Red2019BS['Int_Barthel']==2.0]
XRdepmset=set(Red2019BS.loc[Red2019BS['Int_Barthel']==2.0].index)
XRdepmset


# In[389]:


len(XRdepmset)


# In[294]:


XRdepm[['B.Vestirse', 'B.Desplaz', 'B.Orina', 'B.Silla', 'B.Escal', 'B.Retrete', 'B.Comer','Int_Barthel']].hist(figsize=[14, 12])

plt.savefig("XRdep2019m.png", bbox_inches='tight', dpi=300)


# In[559]:


from operator import index
XRdeps=Red2019BS.loc[Red2019BS['Int_Barthel']==3.0]
XRdepsset=set(Red2019BS.loc[Red2019BS['Int_Barthel']==3.0].index)
XRdepsset


# In[560]:


XRdeps[['B.Vestirse', 'B.Desplaz', 'B.Orina', 'B.Silla', 'B.Escal', 'B.Retrete', 'B.Comer','Int_Barthel']].hist(figsize=[14, 12])

plt.savefig("XRdep201sm.png", bbox_inches='tight', dpi=300)


# In[310]:


r=Red2019BS.columns[0:-1]
r


# In[313]:


IND=indiscernibility(U, df2019BS)
IND


# In[314]:


len(IND[1])/len(df2019BS)*100


# In[319]:


df2019BS.loc[df2019BS['Int_Barthel']==1]
#len(df2019BS.loc[df2019BS['Int_Barthel']==1])


# In[320]:


df2019BS.loc[df2019BS['Int_Barthel']==2]


# In[321]:


df2019BS.loc[df2019BS['Int_Barthel']==3]


# In[603]:


Red2019BS.iloc[77]


# In[344]:


Red2019BS.iloc[1]


# In[345]:


Red2019BS.loc[1].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[323]:


Red2019BS.iloc[130]


# In[347]:


Red2019BS.loc[130].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[348]:


Red2019BS.iloc[32]


# In[349]:


Red2019BS.loc[32].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[350]:


Red2019BS.iloc[81]


# In[351]:


Red2019BS.loc[81].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[352]:


Red2019BS.iloc[59]


# In[353]:


Red2019BS.loc[59].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[354]:


Red2019BS.iloc[44]


# In[356]:


Red2019BS.loc[44].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[357]:


Red2019BS.iloc[125]


# In[397]:


Red2019BS.loc[160].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[396]:


Red2019BS.iloc[160]


# In[402]:


Red2019BS.loc[82].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[403]:


Red2019BS.iloc[82]


# In[359]:


Red2019BS.iloc[102]


# In[360]:


Red2019BS.loc[102].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[333]:


# import necessary modules
import numpy as np
import matplotlib.pyplot as plt
from math import pi



# obtain df information
categories = list(df2019BS)[1:-1]
values = df2019BS.mean().values.flatten().tolist()
values += values[:1] # repeat the first value to close the circular graph
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]

# define plot
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8),
                        subplot_kw=dict(polar=True))
plt.xticks(angles[:-1], categories, color='grey', size=12)
plt.yticks(np.arange(0, 25, 5), ['0', '5', '10','15','20'],
           color='grey', size=12)
plt.ylim(0, 20)
ax.set_rlabel_position(30)

# draw radar-chart:
for i in range(len(df2019BS)):
    val_c1 = df2019BS.loc[i].drop(['Nombre','Int_Barthel']).values.flatten().tolist()
    val_c1 += val_c1[:1]
    ax.plot(angles, val_c1, linewidth=1, linestyle='solid',
            label=df2019BS.loc[i]["Nombre"])
    ax.fill(angles, val_c1, alpha=0.4)

# add legent and show plot
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()


# In[ ]:


df2019BS.loc[11].drop(['Nombre','Int_Barthel']).values.flatten().tolist()


# In[ ]:


[10, 15, 10, 15, 10, 10, 10]
[10, 15, 10, 15, 5, 10, 10]
[10, 15, 5, 15, 10, 10, 10]
[10, 15, 10, 15, 10, 10, 5]
[10, 10, 10, 15, 10, 10, 10]
[10, 15, 10, 10, 10, 10, 10]
[5, 15, 10, 15, 10, 10, 10]
[10, 15, 10, 15, 5, 10, 5]
[10, 10, 10, 15, 5, 10, 10]
[10, 5, 5, 5, 0, 0, 5]
B.Vestirse                         10
B.Desplaz                           5
B.Orina                             5
B.Silla                             5
B.Escal                             0
B.Retrete                           0
B.Comer                             5
Int_Barthel                         3


# In[373]:





# In[376]:


categories=list(df)[1:]
categories


# In[384]:


row


# In[420]:


# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 

    
# Set data
df = pd.DataFrame({
'Dependencia': ['Nula (119)', 'Leve (10)', 'Leve (6)', 'Leve (4))', 'Leve (3)', 'Leve (2)', 'Leve (2)', 'Moderada (3)','Moderada (3)','Severa (1)'],
'B.Vestirse': [10, 10, 10, 10, 10, 10,  5, 10, 10, 10],
'B.Desplaz':  [15, 15, 15, 15, 10, 15, 15, 15, 10,  5],
'B.Orina':    [10, 10,  5, 10, 10, 10, 10, 10, 10,  5],
'B.Silla':    [15, 15, 15, 15, 15, 10, 15, 15, 15,  5],
'B.Escal':    [10,  5, 10, 10, 10, 10, 10,  5,  5,  0],
'B.Retrete':  [10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
'B.Comer':    [10, 10, 10,  5, 10, 10, 10,  5, 10,  5]
})
 
# ------- PART 1: Define a function that do a plot for one line of the dataset!
 
def make_spider( row, title, color):

    # number of variable
    categories=list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(4,4,row+1, polar=True, )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='black', size=14)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([5,10,15], ["5","10","15"], color="black", size=12)
    plt.ylim(0,15)

    # Ind1
    values=df.loc[row].drop('Dependencia').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=16, color=color, y=1.1)

    
# ------- PART 2: Apply the function to all individuals
# initialize the figure
my_dpi=90
plt.figure(figsize=(2800/my_dpi, 2800/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(df.index))
 
# Loop to plot
for row in range(0, len(df.index)):
    make_spider( row=row, title='Dependencia '+df['Dependencia'][row], color=my_palette(row))
    
    
plt.savefig("Redcto2019Radar.png", bbox_inches='tight' , dpi=300)


# In[381]:


# Import the library
import plotly.express as px

# Load the iris dataset provided by the library
#df = 

# Create the chart:
fig = px.parallel_coordinates(
    df, 
    color="species_id", 
    labels={"species_id": "Species","sepal_width": "Sepal Width", "sepal_length": "Sepal Length", "petal_width": "Petal Width", "petal_length": "Petal Length", },
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=2)

# Hide the color scale that is useless in this case
fig.update_layout(coloraxis_showscale=False)

# Show the plot
fig.show()


# In[38]:


df2019BS=df2019[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete',
       'B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina',
       'Int_Barthel']]

dff=df2019BS.loc[df2019BS['Int_Barthel']==1.0]
A=list(dff.index)
B=set(A)
C=list(B)


# In[421]:


low_approximation(IND,C)


# In[422]:


upper_approximation(IND, B)


# In[423]:


INDr=indiscernibility(r,Red2019BS)
INDr


# In[424]:


low_approximation(INDr,C)


# In[425]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

a = np.random.randint(0,5, size=(15,10))
print(a)
colors = ['black', 'yellow', 'red','blue']
boundaries = np.arange(1,5)-0.5
cmap = matplotlib.colors.ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(boundaries, len(colors))

plt.imshow(a, cmap=cmap, norm=norm)

cb = plt.colorbar(ticks=np.arange(1,4))

plt.show()


# In[530]:


#carga los datos de los archivos de excel con los resultados del test de Barthel
df2021 = pd.read_excel('2021BARTH.xlsx')
df2021 = df2021.dropna() #quita las filas que tengan NaN en algun valor
df2021


# In[531]:


Listadf2021=df2021['Nombre'].tolist() # crea una lista con los usuarios de 2021
Setdf2021=set(Listadf2021) # crea un conjunto a partir de la lista de usuarios de 2021


# In[532]:


venn2([Setdf2019, Setdf2021], set_labels = ('Base de datos de 2019', 'Base de datos de 2021'), set_colors=('red','blue'))

plt.show()


# In[534]:


Recurrentes2019a2021=list(Setdf2019.intersection(Setdf2021))

A=[]

def createList(n):
    return np.arange(0, n, 1)


# In[529]:


Recurrentes2019a2021=list(Setdf2019.intersection(Setdf2021))

A=[]

def createList(n):
    return np.arange(0, len(Recurrentes2019a2021), 1)

for i in createList(len(Recurrentes2019a2021)-1): 
  for idx in df2019.index[df2019['Nombre']== Recurrentes2019a2021[i]]:
    A.append(idx)

B=[]
for i in createList(len(Recurrentes2019a2021)-1): 
  for idx in df2021.index[df2021['Nombre']== Recurrentes2019a2021[i]]:
    B.append(idx)



dfRecurrentes20192021a=df2019.loc[A]
dfRecurrentes20192021b=df2021.loc[B]


with pd.ExcelWriter('dfRecurrentes2019y2021.xlsx', engine='xlsxwriter') as writer:
    dfRecurrentes20192021a.to_excel(writer, sheet_name='Año 2019')
    dfRecurrentes20192021b.to_excel(writer, sheet_name='Año 2021')
    


#files.download('dfRecurrentes2019y2021.xlsx')


# In[426]:


import pingouin as pg

pg.cronbach_alpha(data=df2021[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete',
       'B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina']])


# In[427]:


df2021M=df2021.loc[df2021['Sexo']==1]
df2021M


# In[428]:


hist = df2021M['Edad'].hist(bins=10)
plt.xlabel("Edad (años)")
 
# Label for y-axis
plt.ylabel("Número de individuos")
plt.savefig("Mujeres2021.png", bbox_inches='tight', dpi=300)


# In[429]:


import pingouin as pg

pg.cronbach_alpha(data=df2021M[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete',
       'B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina']])


# In[430]:


df2021H=df2021.loc[df2021['Sexo']==2]
df2021H


# In[431]:


hist = df2021H['Edad'].hist(bins=7)
plt.xlabel("Edad (años)")
 
# Label for y-axis
plt.ylabel("Número de individuos")
plt.savefig("Hombres2021.png", bbox_inches='tight', dpi=300)


# In[432]:


import pingouin as pg

pg.cronbach_alpha(data=df2021H[['B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete',
       'B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina']])


# In[492]:


M2021E60=df2021M.loc[(df2021M['Edad']<=60)]
M2021E60


# In[499]:


M2021E60.describe()


# In[493]:


M2021E70=df2021M.loc[((df2021M['Edad']<=70) & (df2019['Edad']>60))]
M2021E70


# In[500]:


M2021E70.describe()


# In[494]:


M2021E80=df2021M.loc[((df2021M['Edad']<=80) & (df2019['Edad']>70))]
M2021E80


# In[498]:


M2021E80.describe()


# In[495]:


M2021E80p=df2021M.loc[((df2021M['Edad']>80))]
M2021E80p


# In[497]:


M2021E80p.describe()


# In[510]:


H2021E60=df2021H.loc[(df2021H['Edad']<=60)]
H2021E60


# In[504]:


H2021E70=df2021H.loc[((df2021H['Edad']<=70) & (df2021H['Edad']>60))]
H2021E70


# In[509]:


H2021E70.describe()


# In[505]:


H2021E80=df2021H.loc[((df2021H['Edad']<=80) & (df2021H['Edad']>70))]
H2021E80


# In[508]:


H2021E80.describe()


# In[506]:


H2021E80p=df2021H.loc[(df2021H['Edad']>80)]
H2021E80p


# In[507]:


H2021E80p.describe()


# In[537]:


df2021BS=df2021[['Nombre','B.Comer', 'B.Silla', 'B.Aseo', 'B.Retrete',
       'B.Ducha', 'B.Desplaz', 'B.Escal', 'B.Vestirse', 'B.Heces', 'B.Orina',
       'Int_Barthel']]


# In[538]:


df2021BS


# In[539]:


df2021BS.columns[0:-1]


# In[541]:


#a program to find the lower approximation of a feature/ set of features
#mohana palaka
#2019

import pandas as pd
import numpy as np
import time

def indiscernibility(attr, table):

	u_ind = {}	#an empty dictionary to store the elements of the indiscernibility relation (U/IND({set of attributes}))
	attr_values = []	#an empty list to tore the values of the attributes

	for i in (table.index):

		attr_values = []
		for j in (attr):

			attr_values.append(table.loc[i, j])	#find the value of the table at the corresponding row and the desired attribute and add it to the attr_values list

		#convert the list to a string and check if it is already a key value in the dictionary
		key = ''.join(str(k) for k in (attr_values))

		if(key in u_ind):	#if the key already exists in the dictionary
			u_ind[key].add(i)

		else:	#if the key does not exist in the dictionary yet
			u_ind[key] = set()
			u_ind[key].add(i)

	return list(u_ind.values())



def lower_approximation(R, X):	#We have to try to describe the knowledge in X with respect to the knowledge in R; both are LISTS OS SETS [{},{}]

	l_approx = set()	#change to [] if you want the result to be a list of sets

	#print("X : " + str(len(X)))
	#print("R : " + str(len(R)))

	for i in range(len(X)):
		for j in range(len(R)):

			if(R[j].issubset(X[i])):
				l_approx.update(R[j])	#change to .append() if you want the result to be a list of sets

	return l_approx



def gamma_measure(describing_attributes, attributes_to_be_described, U, table):	#a functuon that takes attributes/features R, X, and the universe of objects

	f_ind = indiscernibility(describing_attributes, table)
	t_ind = indiscernibility(attributes_to_be_described, table)

	f_lapprox = lower_approximation(f_ind, t_ind)

	return len(f_lapprox)/len(U)

	#return mod_l_approx(l_approx)/len(U)


def quick_reduct(C, D, table):	#C is the set of all conditional attributes; D is the set of decision attributes

	reduct = set()

	gamma_C = gamma_measure(C, D, table.index, table)
	print(gamma_C)
	gamma_R = 0

	while(gamma_R < gamma_C):

		T = reduct

		for x in (set(C) - reduct):

			feature = set()	#creating a new set to hold the currently selected feature
			feature.add(x)

			print(feature)

			new_red = reduct.union(feature)	#directly unioning x separates the alphabets of the feature...

			gamma_new_red = gamma_measure(new_red, D, table.index, table)
			gamma_T = gamma_measure(T, D, table.index, table)

			if(gamma_new_red > gamma_T):

				T = reduct.union(feature)
				print("added")

		reduct = T

		#finding the new gamma measure of the reduct

		gamma_R = gamma_measure(reduct, D, table.index, table)
		print(gamma_R)

	return reduct


t1 = time.time()

#final_reduct = quick_reduct(audio.columns[0:-1], [audio.columns[-1]], audio)
#final_reduct = quick_reduct(mushroom.columns[1:], [mushroom.columns[0]], mushroom)

final_reduct=quick_reduct(df2021BS.columns[1:-1],[df2021BS.columns[-1]],df2021BS)
print("Serial took : ", str(time.time() - t1))
print(final_reduct)
'''
w_ind = indiscernibility(['weak'], my_table)
d_ind = indiscernibility(['flu'], my_table)
w_gamma = gamma_measure(['weak'], ['flu'], my_table.index)
print(w_gamma)
'''     


# In[542]:


def low_approximation(R, X):	#We have to try to describe the knowledge in X with respect to the knowledge in R; both are LISTS OS SETS [{},{}]

	low_approx = set()	#change to [] if you want the result to be a list of sets

	#print("X : " + str(len(X)))
	#print("R : " + str(len(R)))

	for j in range(len(R)):

			if(R[j].issubset(X)):
				low_approx.update(R[j])	#change to .append() if you want the result to be a list of sets

	return low_approx


# In[543]:


def upper_approximation(R, X):	#We have to try to describe the knowledge in X with respect to the knowledge in R; both are LISTS OS SETS [{},{}]

	u_approx = set()	#change to [] if you want the result to be a list of sets

	#print("X : " + str(len(X)))
	#print("R : " + str(len(R)))

	for j in range(len(R)):

			if(R[j].intersection(X)!=set()):
				u_approx.update(R[j])	#change to .append() if you want the result to be a list of sets

	return u_approx


# In[544]:


df2021BS.columns[1:-1]


# In[547]:


Red2021BS=df2021[['Nombre','Sexo','Edad','B.Vestirse', 'B.Desplaz', 'B.Orina', 'B.Silla', 'B.Escal', 'B.Retrete', 'B.Comer','Int_Barthel']]
Red2021BS


# In[ ]:


r=Red2019BS.columns[0:-1]
r


# In[548]:


from operator import index
XRindep=Red2021BS.loc[Red2021BS['Int_Barthel']==0.0]
XRindepset=set(Red2021BS.loc[Red2021BS['Int_Barthel']==0.0].index)
XRindepset


# In[549]:


XRindep[['B.Vestirse', 'B.Desplaz', 'B.Orina', 'B.Silla', 'B.Escal', 'B.Retrete', 'B.Comer','Int_Barthel']].hist(figsize=[14, 12])

plt.savefig("XRindep2019.png", bbox_inches='tight', dpi=300)


# In[552]:


from operator import index
XRdepl=Red2021BS.loc[Red2021BS['Int_Barthel']==1.0]
XRdeplset=set(Red2021BS.loc[Red2021BS['Int_Barthel']==1.0].index)
XRdeplset


# In[553]:


XRdepl[['B.Vestirse', 'B.Desplaz', 'B.Orina', 'B.Silla', 'B.Escal', 'B.Retrete', 'B.Comer','Int_Barthel']].hist(figsize=[14, 12])

plt.savefig("XRdep2019l.png", bbox_inches='tight', dpi=300)


# In[554]:


from operator import index
XRdepm=Red2021BS.loc[Red2021BS['Int_Barthel']==2.0]
XRdepmset=set(Red2021BS.loc[Red2021BS['Int_Barthel']==2.0].index)
XRdepmset


# In[555]:


XRdepm[['B.Vestirse', 'B.Desplaz', 'B.Orina', 'B.Silla', 'B.Escal', 'B.Retrete', 'B.Comer','Int_Barthel']].hist(figsize=[14, 12])

plt.savefig("XRdep2019m.png", bbox_inches='tight', dpi=300)


# In[556]:


from operator import index
XRdeps=Red2021BS.loc[Red2021BS['Int_Barthel']==3.0]
XRdepsset=set(Red2021BS.loc[Red2021BS['Int_Barthel']==3.0].index)
XRdepsset


# In[557]:


XRdeps[['B.Vestirse', 'B.Desplaz', 'B.Orina', 'B.Silla', 'B.Escal', 'B.Retrete', 'B.Comer','Int_Barthel']].hist(figsize=[14, 12])

plt.savefig("XRdep2021s.png", bbox_inches='tight', dpi=300)


# In[561]:


U=df2021BS.columns[1:-1]
U


# In[563]:


IND=indiscernibility(U, df2021BS)
IND


# In[568]:


r=Red2021BS.columns[3:-1]
r


# In[569]:


indiscernibility(r, df2021BS)


# In[595]:


len(indiscernibility(r, df2021BS)[2])


# In[585]:


Red2021BS.loc[0]


# In[587]:


Red2021BS.loc[1]


# In[588]:


Red2021BS.iloc[17]


# In[589]:


Red2021BS.iloc[114]


# In[590]:


Red2021BS.iloc[9]


# In[591]:


Red2021BS.iloc[12]


# In[597]:


Red2021BS.iloc[77]


# In[604]:


Red2019BS.iloc[77]


# In[577]:


Red2021BS.loc[0].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[578]:


Red2021BS.iloc[1].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[579]:


Red2021BS.iloc[17].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[580]:


Red2021BS.iloc[114].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[598]:


Red2021BS.iloc[9].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[599]:


Red2021BS.iloc[12].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[600]:


Red2021BS.iloc[77].drop(['Nombre','Sexo','Edad','Int_Barthel']).values.flatten().tolist()


# In[ ]:


[10, 15, 10, 15, 10, 10, 10]
[10, 15, 10, 15, 5, 10, 10]
[10, 15, 5, 15, 10, 10, 10]
[10, 15, 10, 15, 0, 10, 10]
[10, 10, 10, 15, 5, 10, 10]
[10, 10, 10, 10, 5, 10, 10]
[5, 10, 5, 5, 5, 5, 10]


# In[602]:


# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
 

    
# Set data
df = pd.DataFrame({
'Dependencia': ['Nula (106)', 'Leve (1)', 'Leve (2)', 'Moderado (28)', 'Moderado (2)', 'Moderado (2)','Severo (1)'],
'B.Vestirse': [10, 10, 10, 10, 10, 10, 5],
'B.Desplaz':  [15, 15, 15, 15, 10, 10,10],
'B.Orina':    [10, 10,  5, 10, 10, 10, 5],
'B.Silla':    [15, 15, 15, 15, 15, 10, 5],
'B.Escal':    [10,  5, 10,  0,  5,  5, 5],
'B.Retrete':  [10, 10, 10, 10, 10, 10, 5],
'B.Comer':    [10, 15, 10,  5, 15, 10,10]
})
 
# ------- PART 1: Define a function that do a plot for one line of the dataset!
 
def make_spider( row, title, color):

    # number of variable
    categories=list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(4,4,row+1, polar=True, )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='black', size=14)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([5,10,15], ["5","10","15"], color="black", size=12)
    plt.ylim(0,15)

    # Ind1
    values=df.loc[row].drop('Dependencia').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    plt.title(title, size=16, color=color, y=1.1)

    
# ------- PART 2: Apply the function to all individuals
# initialize the figure
my_dpi=90
plt.figure(figsize=(2800/my_dpi, 2800/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(df.index))
 
# Loop to plot
for row in range(0, len(df.index)):
    make_spider( row=row, title='Dependencia '+df['Dependencia'][row], color=my_palette(row))
    
    
plt.savefig("Redcto2021Radar.png", bbox_inches='tight' , dpi=300)


# In[558]:


from operator import index
XRdept=Red2021BS.loc[Red2021BS['Int_Barthel']==4.0]
XRdeptset=set(Red2021BS.loc[Red2021BS['Int_Barthel']==4.0].index)
XRdeptset


# In[43]:


#carga los datos de los archivos de excel con los resultados del test de Barthel
dfEdades=pd.read_excel('Edades.xlsx')
dfEdades.head() #muestra las primeras cinco filas de la base de datos


# In[44]:


dfEdades['Nombre Completo']= dfEdades['Nombres'] + dfEdades['Apellidos'] #combina la columna de Nombres y la de Apellidos
dfEdades # muestra el dataframe resultante


# In[45]:


del dfEdades['Apellidos'] #elimina las siguientes filas porque no son necesarias
del dfEdades['Nombres']
del dfEdades['Sexo']
dfEdades # muestra el dataframe resultante


# In[46]:


# Intercambia el orden de las columnas
DBEdades=dfEdades[['Nombre Completo', 'Edad']]
DBEdades


# In[47]:


ListaDBEdades=DBEdades['Nombre Completo'].tolist() #Toma la columna de nombres 
# del archivo de usuarios con edades registradas y crea una lista


# In[48]:


SetDBEdades=set(ListaDBEdades) #convierte la lista de usuarios cuya edad está registrada en un conjunto


# In[49]:


#carga los datos de los archivos de excel con los resultados de diferentes test para el año 2018
df2018=pd.read_excel('2018.xlsx')
df2018


# In[50]:


del df2018['PuntajeZ'] #quita la fila de puntaje Z, ya que no se tienen datos
del df2018['Marcha'] #quita la fila de Marcha, ya que no se tienen datos

# Se muestra la base depurada, en la que ya se han eliminado aquellos datos con
# NaN, como las columnas PuntajeZ y Marcha tienen solo NaN, entonces se 
# eliminaron, ya que no aportan información al respecto.

df2018 = df2018.dropna() #quita las filas que tengan NaN en algun valor
df2018


# In[51]:


df2018['Nombre Completo']= df2018['Nombres'] + df2018['Apellidos'] 
del df2018['Apellidos']
del df2018['Nombres']
df2018 #combina las columnas de nombres y apellidos en una sola llamada "Nombre Completo"
# y elimina las columnas individuales.


# In[52]:


# Cambia el orden de las columnas
df2018[['Nombre Completo', 'Sexo', 'MNA', 'Prom_Fuer','Proteinas','BARTHEL', 'Int_BARTHEL']]


# In[53]:


Listadf2018=df2018['Nombre Completo'].tolist()
Setdf2018=set(Listadf2018)


# In[54]:


venn2([Setdf2018, SetDBEdades], set_labels = ('Base de datos de 2018', 'Usuarios con edad registrada'), set_colors=('red','blue'))
# crea un diagrama de Venn en donde podemos ver los usuarios que tienen en común la base de datos de 2018 y la de edades registradas
plt.show()


# In[55]:


ddf2018 = pd.merge(left=df2018,right=DBEdades, how="inner",on="Nombre Completo")
ddf2018 # Combina las bases de datos de 2018 con la de usuarios con edad registrada, dejando solo los que tienen en comun
# es decir, la intersección vista en el diagrama de Venn.


# In[56]:


BD2018=ddf2018[['Nombre Completo','Edad','Sexo', 'MNA', 'Prom_Fuer','Proteinas','BARTHEL', 'Int_BARTHEL']]
BD2018 # Cambia el orden de las columnas y se guarda como una base de datos nueva.


# In[58]:


# writing to Excel
BD2018Depurada = pd.ExcelWriter('BD2018Depurada.xlsx')
BD2018.to_excel(BD2018Depurada)
BD2018Depurada.save()


# In[60]:


sn.displot(BD2018['Edad'])


# In[62]:


BD2018.hist(figsize=[14, 12])
plt.savefig("Xdepm2019.png", bbox_inches='tight', dpi=300)


# In[63]:


altcat.catplot(BD2018,
               height=350,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('MNA:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre Completo"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip("BARTHEL"),
                             alt.Tooltip("Prom_Fuer"),
                             ],
                             color=alt.Color('Sexo', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()


# In[64]:


altcat.catplot(BD2018,
               height=350,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('Prom_Fuer:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre Completo"),
                             alt.Tooltip("Edad"), 
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip("BARTHEL"),
                             alt.Tooltip("MNA"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()


# In[65]:


altcat.catplot(BD2018,
               height=350,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('Proteinas:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre Completo"),
                             alt.Tooltip("Edad"),
                             alt.Tooltip("Prom_Fuer"),
                             alt.Tooltip("BARTHEL"),
                             alt.Tooltip("MNA"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()


# In[66]:


altcat.catplot(df2018,
               height=350,
               width=450,
               mark='point',
               box_mark=dict(strokeWidth=2, opacity=0.6),
               whisker_mark=dict(strokeWidth=2, opacity=0.9),
               encoding=dict(x=alt.X('Sexo:N', title=None),
                             y=alt.Y('BARTHEL:Q',scale=alt.Scale(zero=False)),
                             tooltip=[alt.Tooltip("Nombre Completo"),
                             alt.Tooltip("Proteinas"),
                             alt.Tooltip("Prom_Fuer"),
                             alt.Tooltip("MNA"),
                             ],
                             color=alt.Color('Sexo:N', legend=None)),
               transform='jitterbox',
              jitter_width=0.5).interactive()


# In[68]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Creamos una correlación desde un dataset D
corr = BD2018.corr()

# Dibujamos nuestro gráfico
sns.heatmap(corr)
plt.show()


# In[67]:


selection = alt.selection_multi(fields=['Sexo'], bind='legend')
chart = alt.Chart(BD2018).mark_circle(size=50).encode(
    x='Edad',
    color='Sexo',
    tooltip=[alt.Tooltip("Nombre Completo"),
    alt.Tooltip("Edad"),
    alt.Tooltip("Proteinas"),
    alt.Tooltip("BARTHEL"),
    alt.Tooltip("Prom_Fuer"),
    ],
    opacity=alt.condition(selection, alt.value(1), alt.value(0))
).properties(
    height=400, width=500
).add_selection(
    selection
).interactive()
chart.encode(y='Prom_Fuer') | chart.encode(y='MNA') | chart.encode(y='BARTHEL') | chart.encode(y='Proteinas')


# In[69]:


Hombres2018=BD2018.loc[BD2018['Sexo']=="Mas"]


# In[70]:


Hombres2018.describe()


# In[71]:


Mujeres2018=BD2018.loc[BD2018['Sexo']=="Fem"]


# In[72]:


Mujeres2018.describe()


# In[73]:


BDH2018E60=BD2018.loc[(BD2018['Edad']<=60) & (BD2018['Sexo']=='Mas')]
BDH2018E70=BD2018.loc[(BD2018['Edad']<=70) & (BD2018['Edad']>60) & (BD2018['Sexo']=='Mas')]
BDH2018E80=BD2018.loc[(BD2018['Edad']<=80) & (BD2018['Edad']>70) & (BD2018['Sexo']=='Mas')]
BDH2018E80p=BD2018.loc[(BD2018['Edad']>80) & (BD2018['Sexo']=='Mas')]


# In[74]:


BDH2018E60


# In[75]:


BDH2018E70


# In[76]:


BDH2018E80


# In[77]:


BDH2018E80p


# In[78]:


BDM2018E60=BD2018.loc[(BD2018['Edad']<=60) & (BD2018['Sexo']=='Fem')]
BDM2018E70=BD2018.loc[(BD2018['Edad']<=70) & (BD2018['Edad']>60) & (BD2018['Sexo']=='Fem')]
BDM2018E80=BD2018.loc[(BD2018['Edad']<=80) & (BD2018['Edad']>70) & (BD2018['Sexo']=='Fem')]
BDM2018E80p=BD2018.loc[(BD2018['Edad']>80) & (BD2018['Sexo']=='Fem')]


# In[79]:


BDM2018E60


# In[81]:


BDM2018E70


# In[82]:


BDM2018E80


# In[83]:


BDM2018E80p


# In[84]:


DBEdades2019=DBEdades # crea una nueva base datos (a la que sumaremos un año para ajustarlo a 2019)
DBEdades2019


# In[85]:


DBEdades2019['Edad']=DBEdades2019['Edad']+1 #Suma un año a todas las edades de la base de datos
DBEdades2019


# In[86]:


#carga la base de datos de 2019
df2019=pd.read_excel('2019.xlsx')
df2019


# In[87]:


# Se muestra la base depurada, en la que ya se han eliminado aquellos datos con
# NaN, A diferencia de 2018, en 2019 su habia valores para las pruebas de 
# Puntaje Z y Marcha.
df2019 = df2019.dropna()
df2019


# In[88]:


df2019 = df2019.dropna()
print("DataFrame después de eliminar filas con NaN")
df2019


# In[89]:


df2019['Nombre Completo']=df2019['Nombres'] + df2019['Apellidos']
del df2019['Apellidos']
del df2019['Nombres']
df2019 # Combinas las columnas de nombre y elimina las cols de "Nombres" y "Apellidos"


# In[90]:


#Reordena las columnas y fuarda la nueva base datos como una nueva variable
ddf2019=df2019[['Nombre Completo', 'Sexo', 'MNA', 'Prom_Fuer','Proteinas','Marcha','PuntajeZ','BARTHEL', 'Int_BARTHEL']]


# In[91]:


ddf2019 = pd.merge(left=ddf2019,right=DBEdades2019, how="inner",on="Nombre Completo")
ddf2019 # Crea una lista combinada con usuarios de 2019 y edad registrada


# In[92]:


BD2019=ddf2019[['Nombre Completo','Edad','Sexo', 'MNA', 'Prom_Fuer','Proteinas', 'Marcha', 'PuntajeZ', 'BARTHEL', 'Int_BARTHEL']]
BD2019 # Crea la base de datos de 2019, con las columnas ya reordenadas


# In[93]:


Listadf2019=df2019['Nombre Completo'].tolist() # crea una lista con los usuarios de 2019
Setdf2019=set(Listadf2019) # crea un conjunto a partir de la lista de usuarios de 2019


# In[ ]:




