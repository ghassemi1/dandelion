from __future__ import division, print_function
import os
import io
import numpy as np
import tensorflow as tf
import PIL.Image as Image

# Flask utils
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

####################################################################################3
SAVED_MODEL_DIR = os.getcwd() + "/saved_model" # MODEL_PATH

TEST_IMAGES_DIR = os.getcwd() + "/image_test"

class_names=np.array(['Abutilon', 'Amarante_a_racine_rouge', 'Chiendent', 'Digitaire_astringente', 'Digitaire_sanguine', 'Echinochloa_pied_de_coq',
                             'Galinsoga_cilie', 'Laiteron_des_champs', 'Liseron_des_champs', 'Luzerne', 'Morelle_noire_de_l_Est', 'Morelle_poilue',
                              'Moutarde_des_champs', 'Moutarde_des_oiseaux', 'Ortie_royale', 'Panic_capillaire', 'Panic_d_automne_genicule', 'Petite_herbe_a_poux',
                               'Phragmite_commun', 'Pissenlit', 'Prele_des_champs', 'Radis_sauvage', 'Renouee_liseron', 'Renouee_persicaire', 'Setaire_geante',
                                'Setaire_glauque', 'Setaire_verte', 'Souchet_comestible', 'Spargoute_des_champs', 'Stellaire_moyenne', 'Tabouret_des_champs',
                                 'Vesce_jargeau'])
####################################################################################

# # Load your trained model
model = tf.keras.models.load_model(SAVED_MODEL_DIR)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print('request.files: ',request.files)
    if request.method == "POST":
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error": "no file"})
        
        try:
            image_bytes = file.read()
            # print('image_bytes: ', type(image_bytes))
            pillow_img = Image.open(io.BytesIO(image_bytes))
            # print('pillow_img: ', type(pillow_img))

            resize_image= pillow_img.resize((224,224))
            array_image = np.array(resize_image)/255

            predicted = model.predict(np.array([array_image]))
            predicted_max_index = np.argmax(predicted, axis=-1)
            predicted_label_batch = class_names[predicted_max_index]

            conf=np.max(predicted, axis=-1)*100

            confidence = round(conf[0], 2)
            result= str(predicted_label_batch[0]) + '    ' + '(Confidence:    ' + str(confidence) + ' % ' + ')'
            return result 
        except Exception as e:
            return jsonify({"error": str(e)})
    return None # "OK"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
    # app.run(debug=True) ## Default

