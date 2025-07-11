import streamlit as st
import pandas as pd
import joblib

pipeline_path = "artefacts/preprocessor/preprocessor.pkl"
model_path = "artefacts/model/svc.pkl"
encoder_path = "artefacts/preprocessor/labelencoder.pkl"
with open(pipeline_path, 'rb') as file1:
    print(file1.read(100))

    try:
        pipeline = joblib.load(pipeline_path)
        print('Pipeline cargada')
    except Exception as e:
        print(f'Error cargando el pipeline {e}')
    
    with open(model_path, 'rb') as file2:
        print(file1.read(100))

    try:
        model = joblib.load(model_path)
        print('SVC cargada')
    except Exception as e:
        print(f'Error cargando el SVC {e}')

    with open(encoder_path, 'rb') as file3:
        print(file1.read(100))

    try:
        Encoder = joblib.load(encoder_path)
        print('Encoder cargada')
    except Exception as e:
        print(f'Error cargando el Encoder {e}')

st.title("WebAPP de Machine Learning")
st.header("Ingreso de los datos")

col1, col2, col3 = st.columns(3)

with col1:

    battery_power = st.slider("Poder de la bateria (mAh)", min_value=500, max_value=2000, value=800)
    clock_speed = st.slider("Velocidad del cpu", min_value=0.5, max_value=3.0)
    fc = st.slider("Camara frontal (Mpx)", min_value=0, max_value=19, step=2)
    int_memory = st.slider("Memoria interna (GB)", min_value=2, max_value=62, value=32)
    px_height = st.slider(
        "Resolucion de la pantalla (altura en Px)", min_value=100,max_value=2000
    )


with col2:

    m_dep = st.slider("Grosor del telefono", min_value=0.1, max_value=1.0)
    mobile_wt = st.slider("Peso del telefono", min_value=100, max_value=1000)
    n_cores = st.slider("Numero de nucleos", min_value=1, max_value=10)
    pc = st.slider("Camara trasera MP", min_value=1, max_value=19)
    px_width = st.slider("Resolucion de la pantalla (ancho en PX)", min_value=100, max_value=2000)

with col3:

    ram = st.slider("Memoria RAM", min_value=256, max_value=4000)
    sc_h = st.slider("Altura de la pantalla cm", min_value=10, max_value=12)
    sc_w = st.slider("Ancho de la pantalla", min_value= 0, max_value=18)
    talk_time = st.slider("Duracion de la bateria en uso maximo (Hrs)", min_value=2, max_value=20)

st.divider()
col4, col5, col6 = st.columns(3)

with col4:
    blue = st.selectbox('Tiene bluetooth?' , [0, 1])
    three_g = st.selectbox('Tiene 3G?', [0,1])

with col5:
    dual_sim = st.selectbox('Tiene Dual Sim?' , [0, 1])
    touch_screen = st.selectbox('Tiene Pantalla Tactil', [0,1])

with col6:
    four_g = st.selectbox('Tiene 4G?' , [0, 1])
    wifi = st.selectbox('Tiene Wifi?', [0,1])

st.divider()

if st.button('Predict'):

    input_data = pd.DataFrame(
        {
            'battery_power': [battery_power], 
            'blue': [blue],
            'clock_speed': [clock_speed],
            'dual_sim': [dual_sim],
            'fc': [fc],
            'four_g': [four_g],
            'int_memory': [int_memory],
            'm_dep': [m_dep],
            'mobile_wt': [mobile_wt],
            'n_cores': [n_cores],
            'pc': [pc],
            'px_height': [px_height],
            'px_width': [px_width],
            'ram': [ram],
            'sc_h': [sc_h],
            'sc_w': [sc_w],
            'talk_time': [talk_time],
            'three_g': [three_g],
            'touch_screen': [touch_screen],
            'wifi': [wifi]
        }
    )
    st.dataframe(input_data)

    pipelined_data = pipeline.transform(input_data)

    prediction = model.predict(pipelined_data)

#st.write(prediction)
    if prediction[0] == 0:
        st.success('El precio del dispositivo es bajo')
    elif prediction[0] == 1:
        st.success('El precio del dispositivo es medio')
    elif prediction[0] == 2:
        st.success('El precio del dispositivo es alto')
    elif prediction[0] == 3:
        st.success('El precio del dispositivo es muy alto')