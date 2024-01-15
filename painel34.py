import model
import prevel_model
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st
import projeto


def painel34(df, data_selecionada, hora_selecionada):
    # Criando dois painéis na mesma linha
    col3, col4 = st.columns(2)

    with col3:
        st.write("# Painel 3")
        # Adicione o conteúdo do Painel 3 aqui

        if not df.empty and 'ds' in df.columns:
            """
            ### Passo 2: Modelo
            """
            model.modelo(df, data_selecionada, hora_selecionada)
            st.markdown(
                """
                <style>
                    .reportview-container .main .block-container {
                        max-width: 80%;
                        justify-content: center;
                    }
    
                    .custom-container {
                        border: 2px solid black;
                        border-radius: 5px;
                        padding: 20px;
                        width: 80%;
                        text-align: justify;
                        margin: 20px 0;
                    }
    
                    .nested-container {
                        border: 2px solid black;
                        border-radius: 5px;
                        padding: 10px;
                        text-align: justify;
                        margin: 10px 0;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
            # Definindo guias
            tabs = ["Conceitos", "MAE", "MAPE", "RMSE", "Acurácia"]
            selected_tab_painel34 = st.sidebar.radio("Escolha uma guia:", tabs, key="unique_key_painel34")


            # Conteúdo das guias
            tab_contents = {
                "Conceitos": """
                    <div class="nested-container">
                    <h6><strong>Conceitos</strong></h6>
                    <p><strong>MAE (Mean Absolute Error)</strong>: Representa a média das diferenças absolutas entre as previsões e os valores reais. Indica o quão perto as previsões estão dos valores reais, sem considerar a direção do erro.</p>
                    ...
                    </div>
                    """,
                "MAE": """
                    <div class="nested-container">
                        <h6>MAE</h6>
                        <p>O valor do MAE, como qualquer métrica de erro, depende muito do contexto e da escala dos dados que você está considerando. No contexto do mercado de ações, um MAE de 12,06 pontos pode ser considerado alto ou baixo dependendo do valor médio dos índices ou ativos que você está analisando.</p>
                        ...
                    </div>
                    """,
                "MAPE": """
                    <div class="nested-container">
                        <h6><strong>MAPE</strong></h6>
                        <p>Para a bolsa de valores, 11,65% é um valor razoável?</p>
                        ...
                    </div>
                    """,
                "RMSE": """
                    <div class="nested-container">
                        <h6><strong>RMSE</strong></h6>
                        <p>Em muitos casos envolvendo previsão na bolsa de valores, um RMSE de 11,65 pode ser considerado alto, especialmente se estiver lidando com a previsão de preços de ações individuais ou ativos específicos...</p>
                        ...
                    </div>
                    """,
                "Acurácia": """
                    <div class="nested-container">
                        <h6><strong>Acurácia</strong></h6>
                        <p>Em modelos de séries temporais, o conceito de "acurácia" não é tão direto quanto em modelos de classificação, onde se pode calcular a precisão de forma direta...</p>
                        ...
                    </div>
                    """
            }

            # Exibindo o conteúdo da guia selecionada
            st.markdown(tab_contents[selected_tab], unsafe_allow_html=True)
    with col4:
        st.write("# Painel 4")
        # Adicione o conteúdo do Painel 3 aqui

        """
        ### Passo 3: Previsão no Intervalo 01/01/2024 a 31/01/2024  
        """
        flag = False
        data = st.slider('Data', 1, 31, 1)
        if data <= 9:
            data2 = '2024-01-0' + str(data)
        else:
            data2 = '2024-01-' + str(data)

        btn = st.button("Previsão")

        if btn:
            x = prevel_model.prevendo(projeto.df, data2, flag)
            if x is None:
                st.write(f"A data {data2} não está disponível nas previsões ou é feriado/final de semana.")
            else:
                rounded_x = round(x, 3)
                st.write(f"Valor previsto para {data2}: {rounded_x}")
        flag = True
        prevel_model.prevendo(projeto.df, data2)
