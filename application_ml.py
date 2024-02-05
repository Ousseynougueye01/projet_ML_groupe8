# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 21:18:34 2024

@author: ousse
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
from plotly.subplots import make_subplots
from ydata_profiling import ProfileReport
from yellowbrick.regressor import ResidualsPlot
import streamlit as st

# Return a new path pointing to the current working directory
HOME_DIR = "C:/Users/ousse/OneDrive/Bureau/Projet_ML_Groupe8"
print(HOME_DIR)
# create a variable for data directory
DATA_DIR = Path(HOME_DIR, "data")
print(DATA_DIR)
print(f"Work directory: {HOME_DIR} \nData directory: {DATA_DIR}")

# you must put the CSV file 2016_Building_Energy_Benchmarking.csv in data directory, cf above cell
data = pd.read_csv(Path(DATA_DIR, "2016_Building_Energy_Benchmarking.csv"), sep=",")
data_clean = pd.read_csv(Path(DATA_DIR, "data_clean.csv"), sep=",")

# Application avec streamlit

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Analyse exploratoire des données","Une vue sur les bâtiments", "Modélisation"]

page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    
    st.write("### Contexte du projet")
    
    st.write("Le programme d’analyse comparative et de rapport sur l’efficacité énergétique des bâtiments de Seattle (SMC 22.920) exige que les propriétaires de bâtiments non résidentiels et multifamiliaux (20 000 pieds carrés ou plus) suivent leur performance énergétique et en fassent rapport annuellement à la ville de Seattle. Les bâtiments représentent 33 % des principales émissions de Seattle. La politique d’analyse comparative soutient les objectifs de Seattle visant à réduire la consommation d’énergie et les émissions de gaz à effet de serre des bâtiments existants. En 2013, la ville de Seattle a adopté un plan d’action climatique visant à atteindre zéro émission nette de gaz à effet de serre (GES) d’ici 2050. L’analyse comparative annuelle, la production de rapports et la divulgation du rendement des bâtiments sont des éléments fondamentaux de la création d’une plus grande valeur marchande pour l’efficacité énergétique. Conformément à l’ordonnance (125000), à partir du rapport de performance sur la consommation d’énergie de 2015, la ville de Seattle rendra les données de tous les bâtiments de 20 000 pieds carrés et plus disponibles chaque année. Cette mise à jour du mandat d’analyse comparative a été adoptée par le conseil municipal de Seattle le 29 février 2016.")
    

elif page == pages[1]:
    st.write("### Analyse exploratoire")
    
    st.dataframe(data.head())
    
    st.write("Dimensions du dataframe :")
    
    st.write(data.shape)
    st.write(data.describe(include="all"))
    
    # Affichez le graphique MissingNo avec Streamlit
    
    st.write("#### Graphique de données manquantes")
    msno.bar(data)
    st.pyplot()

    st.write("#### Heatmap des données manquantes")
    msno.heatmap(data)
    st.pyplot()
    
    #Distribution TotalGHGEmissions
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
    # Premier sous-graphique : Distribution de TotalGHGEmissions
    sns.histplot(data_clean["TotalGHGEmissions"], color='r', kde=True, ax=axes[0])
    axes[0].set_title('Distribution - TotalGHGEmissions')
    
    # Deuxième sous-graphique : Distribution logarithmique de TotalGHGEmissions
    sns.histplot(np.log(data_clean["TotalGHGEmissions"]), color='b', kde=True, ax=axes[1])
    axes[1].set_title('Distribution - TotalGHGEmissions $log$')
    axes[1].set_xscale('log')
        
    # Utilisez Streamlit pour afficher les graphiques
    st.write("#### Distribution de TotalGHGEmissions avec Streamlit")
    st.pyplot(fig)
    
    #Distribution TotalGHGEmissions
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
    # Premier sous-graphique : Distribution de TotalGHGEmissions
    sns.histplot(data_clean["GHGEmissionsIntensity"], color='r', kde=True, ax=axes[0])
    axes[0].set_title('Distribution - GHGEmissionsIntensity')
    
    # Deuxième sous-graphique : Distribution logarithmique de TotalGHGEmissions
    sns.histplot(np.log(data_clean["GHGEmissionsIntensity"]), color='b', kde=True, ax=axes[1])
    axes[1].set_title('Distribution - GHGEmissionsIntensity $log$')
    axes[1].set_xscale('log')
        
    # Utilisez Streamlit pour afficher les graphiques
    st.write("#### Distribution de GHGEmissionsIntensity")
    st.pyplot(fig)

elif page == pages[2]:
    st.write("### Analyse des bâtiments")
    def plot_map(column):
     fig = px.scatter_mapbox(data_clean,
                            lat='Latitude',
                            lon='Longitude',
                            color=column,
                            hover_name='PrimaryPropertyType')

     fig.update_layout(mapbox_style='open-street-map')

     title = 'map-' + column
     st.plotly_chart(fig, use_container_width=True)

  # Utilisez Streamlit pour appeler la fonction plot_map avec le paramètre 'Neighborhood'
    st.write("### Répartition spatiale des bâtiments")
    plot_map('Neighborhood')
    st.write("### Répartition spatiale de l'intensité d'utilisation de l'énergie")
    plot_map('SiteEUI(kBtu/sf)')
    st.write("### Répartition spatiale des émissions totales de gaz à effet de serre divisées par la superficie brute de la propriété")
    plot_map('GHGEmissionsIntensity')
     
    def visualisation(var, data):
     the_mean = data["GHGEmissionsIntensity"].mean()
     fig = plt.figure(figsize=[18, 7])
     fig.patch.set_facecolor('#E0E0E0')
     fig.patch.set_alpha(0.7)
     plt.title("Distribution des émissions de CO2 selon {}".format(var), size=16)
     sns.boxplot(x=var, y="GHGEmissionsIntensity", data=data, color="#cbd1db", width=0.5, showfliers=False, showmeans=True)
     plt.hlines(y=the_mean, xmin=-0.5, xmax=len(data[var].unique()) - 0.5, color="#6d788b", ls="--", label="Moyenne Globale")

     plt.ylabel("Emissions de CO2", size=14)
     plt.xticks(range(0, len(data[var].unique())), data[var].unique(), rotation=90)
     plt.legend()
     plt.grid()
     st.pyplot(fig)
 # Utilisez Streamlit pour appeler la fonction visualisation avec le paramètre 'LargestPropertyUseType'
    st.write("### Visualisation des émissions de CO2 selon les bâtiments")
    visualisation('LargestPropertyUseType', data_clean)
    visualisation('Neighborhood',data_clean)
    visualisation('PrimaryPropertyType',data_clean)
   