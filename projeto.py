# import plotly.express as px
import textwrap
import warnings
import os
import sys

zip_file_path = "app.zip"

# !pip install matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
# !pip install statsmodels
# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import acf, pacf
import streamlit as st
from prophet.diagnostics import performance_metrics

import model
import prevel_model

# Ignorar os FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# selected_date = '2024-01-25'  # Replace this with the selected date
# selected_time = pd.Timestamp('00:00:00').time()

# from model import modelo
# from prevel_model import prevendo

import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import streamlit as st

# Criação de um componente HTML personalizado para dividir a tela

# Criando dois painéis na mesma linha
col1, col2 = st.columns([2,1])

with col1:
    st.write("# Painel 1")
    # Adicione o conteúdo do Painel 1 aqui

    st.title('📈 Projeção do Índice Bovespa')

    # Importando bibliotecas
    import streamlit as st

    # Estilo para ajustar a largura da área de exibição e justificar o texto
    st.markdown(
        """
        <style>
            .reportview-container .main .block-container {
                max-width: 50%;
                justify-content: center;
            }
    
            .custom-container {
                width: 80%;
                padding: 20px;
                border: 2px solid #333;
                border-radius: 10px;
                margin: 10px 0;
            }
            .custom-container li {
                text-align: justify; /* Alinha o texto à justificação */  
            }
            .custom-container p {
                text-align: justify; /* Alinha o texto à justificação */  
            }
            
        </style>
        """,
        unsafe_allow_html=True
    )

    # Definindo guias
    tabs = ["Visão Geral", "Pontos-chave", "Utilização do Prophet", "Sobre o Autor"]
    selected_tab = st.sidebar.radio("Escolha uma guia:", tabs)

    # Conteúdo das guias
    tab_contents = {
        "Visão Geral": """
        <div class="custom-container">
           <p> Este data app usa a Biblioteca open-source Prophet para automaticamente gerar valores futuros de previsão de um dataset importado. 
            Você poderá visualizar as projeções do índice Bovespa para o período de 01/01/2024 a 31/01/2024 😵.</p>
        </div>
        <style>
            .custom-container p {
                text-align: justify; /* Alinha o texto à justificação */ 
            }
        </style>
        """,
        "Pontos-chave": """
        <div class="custom-container">
          <p>O Prophet tem sido amplamente utilizado em diversas áreas, como previsão de vendas, demanda de produtos, análise financeira, previsão climática e muito mais, devido à sua capacidade de gerar previsões precisas e à sua facilidade de uso. 
            É importante notar que, embora seja uma ferramenta poderosa, a escolha entre modelos depende do contexto específico do problema e da natureza dos dados.</p>
        </div>
        <style>
            .custom-container p {
                text-align: justify; /* Alinha o texto à justificação */
            }
        </style>
        """,
        "Utilização do Prophet": """
        <div class="custom-container">
            <p>A biblioteca Prophet, desenvolvida pelo Facebook, é uma ferramenta popular e poderosa para previsão de séries temporais. Ela foi projetada para simplificar o processo de criação de modelos de previsão, oferecendo aos usuários uma maneira fácil de gerar previsões precisas e de alta qualidade, mesmo sem um profundo conhecimento em séries temporais ou estatística avançada.</p>
            <p>Aqui estão alguns pontos-chave sobre o Prophet:</p>
            <ol>
                <li><strong>Facilidade de Uso:</strong> O Prophet foi desenvolvido para ser acessível e fácil de usar, permitindo que usuários, mesmo sem experiência avançada em séries temporais, possam construir modelos de previsão.</li>
                <li><strong>Componentes Aditivos:</strong> O modelo do Prophet é baseado em componentes aditivos, onde são consideradas tendências anuais, sazonais e efeitos de feriados, além de componentes de regressão.</li>
                <li><strong>Tratamento de Dados Ausentes e Outliers:</strong> O Prophet lida bem com dados ausentes e outliers, reduzindo a necessidade de pré-processamento extensivo dos dados antes da modelagem.</li>
                <li><strong>Flexibilidade:</strong> Permite a inclusão de dados adicionais, como feriados e eventos especiais, para melhorar a precisão das previsões.</li>
                <li><strong>Estimativa Automática de Intervalos de Incerteza:</strong> O Prophet fornece intervalos de incerteza para as previsões, o que é essencial para compreender a confiabilidade dos resultados.</li>
                <li><strong>Implementação em Python e R:</strong> Está disponível tanto para Python quanto para R, ampliando sua acessibilidade para diferentes comunidades de usuários.</li>
                <li><strong>Comunidade Ativa e Documentação Detalhada:</strong> A biblioteca possui uma comunidade ativa de usuários e desenvolvedores, além de uma documentação detalhada e exemplos práticos que ajudam na aprendizagem e na solução de problemas.</li>
            </ol>
        </div>
        """,
        "Sobre o Autor": """
        <div class="custom-container">
            Criado por Henrique José Itzcovici.
            Código disponível em: <a href="https://github.com/Henitz/challenge2" target="_blank">GitHub</a>
        </div>
        """
    }

    # Renderizar conteúdo da guia selecionada
    st.markdown(tab_contents[selected_tab], unsafe_allow_html=True)

with col2:
    st.write("# Painel 2")
    # Adicione o conteúdo do Painel 2 aqui

    """
    ### Passo 1: Importar dados
    """
    df = pd.DataFrame(columns=['Data'])  # Inicializa um DataFrame vazio

    # Adiciona o diretório que contém app.py ao PATH para importações relativas
    # diretorio_app = os.path.dirname(os.path.abspath(__file__))
    # sys.path.append(diretorio_app)
    # from app import pasta_do_zip  # Importa a variável pasta_do_zip de app.py

    import streamlit as st

    # Defina a variável pasta_do_zip aqui
    import os

    pasta_do_zip = os.path.join("c:", "temp_extracted")
    # ou o caminho correto para a sua pasta

    # Original message in Portuguese formatted with HTML

    upload_message = (
        "Importe os dados da série em formato CSV aqui. Posteriormente, as colunas serão nomeadas ds e y. "
        "A entrada de dados para o Prophet sempre deve ser com as colunas: ds e y. "
        "A coluna ds (datestamp) deve ter o formato esperado pelo Pandas, idealmente "
        "YYYY-MM-DD para data ou YYYY-MM-DD HH:MM:SS para timestamp. "
        "A coluna y deve ser numérica e representa a medida que queremos estimar."
    )

    # Use str() to convert the cache key to a string
    key_for_cache = str(os.path.basename(pasta_do_zip)) if os.path.exists(pasta_do_zip) else None

    # File Uploader
    uploaded_file = st.file_uploader(upload_message, type='csv', key=key_for_cache)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if not df.empty and 'Data' in df.columns and 'Último' in df.columns:
            # Renomear colunas para 'ds' e 'y'
            df = df.rename(columns={'Data': 'ds', 'Último': 'y'})
            df['ds'] = pd.to_datetime(df['ds'], format='%d.%m.%Y')

            # Outras operações no DataFrame, como remover colunas indesejadas
            colunas_para_remover = ['Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%']
            df = df.drop(columns=colunas_para_remover)

            st.dataframe(df.style.set_table_attributes('style="height: 50px; overflow: auto;"'))

            if 'ds' in df.columns:
                data_padrao = df['ds'].min()
                hora_padrao = pd.Timestamp('00:00:00').time()  # Hora padrão como 00:00:00
                data_minima = df['ds'].min()  # Data mínima do DataFrame
                data_maxima = df['ds'].max()  # Data máxima do DataFrame

                data_selecionada = st.sidebar.date_input("Selecione uma data", value=data_padrao, min_value=data_minima,
                                                         max_value=data_maxima)
                hora_selecionada = st.sidebar.time_input("Selecione um horário", value=hora_padrao)

                if data_selecionada:
                    # Convertendo a data selecionada para o formato do DataFrame
                    data_selecionada_formatada = pd.to_datetime(data_selecionada).strftime('%Y-%m-%d')

                    # Criar um datetime combinando a data selecionada com a hora padrão
                    data_hora_selecionada = pd.to_datetime(data_selecionada_formatada + ' ' + str(hora_selecionada))

                    # Filtrar o DataFrame com base na data e hora selecionadas
                    df_filtrado = df[df['ds'] == data_hora_selecionada]
                    st.dataframe(df_filtrado)
                else:
                    st.warning("Não há dados para a data selecionada.")
            else:
                st.warning("A coluna 'ds' não está presente no DataFrame.")

        else:
            st.warning("O arquivo não foi carregado corretamente ou não possui as colunas esperadas.")
# Criando dois painéis na mesma linha
col3, col4 = st.columns(2)

with col3:
    st.write("# Painel 3")
    # Adicione o conteúdo do Painel 3 aqui

    if uploaded_file is not None and not df.empty and 'ds' in df.columns:
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
        selected_tab = st.sidebar.radio("Escolha uma métrica:", tabs)

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
        x = prevel_model.prevendo(df, data2, flag)
        if x is None:
            st.write(f"A data {data2} não está disponível nas previsões ou é feriado/final de semana.")
        else:
            rounded_x = round(x, 3)
            st.write(f"Valor previsto para {data2}: {rounded_x}")
    flag = True
    prevel_model.prevendo(df, data)

