from flask import Flask
from flask import send_file,render_template
import os
import cv2 
import json
import pandas as pd
import base64
app = Flask(__name__)

scenegraghs = json.load(open("./static/sceneGraphs/train_sceneGraphs.json"))
df = pd.read_csv("./static/intermediate_questions_samples_25000.csv", dtype={'q_id': str, 'image_id': str, 'entity_id': str})

def encode_image(img, im_type='jpg'):
    success, encoded_img = cv2.imencode('.{}'.format(im_type), img)
    if success:
        return base64.b64encode(encoded_img).decode()
    return ''

@app.route('/image/<image_file>')
def image(image_file):
    
    # path = os.path.join("static/images/" + image_file)
    # return send_file(path, as_attachment=True)
    return render_template("image.html", image_file="images/"+image_file)

@app.route('/sample/<sample_index>')
def bounding_box(sample_index):
    
    example = df.iloc[int(sample_index)]
    # q_id,image_id,original_question,ambiguous_question,ambiguous_question_answer,ambiguous_entity,additional_question,additional_question_answer,entity_id
    q_id = str(example["q_id"])
    image_id = str(example["image_id"])
    original_question = example["original_question"]
    ambiguous_question = example["ambiguous_question"]
    ambiguous_answer = example["ambiguous_question_answer"]
    ambiguous_entity = example["ambiguous_entity"]
    additional_question = example["additional_question"]
    additional_answer = example["additional_question_answer"]
    entity_id = str(example["entity_id"])
    
    image = cv2.imread("./static/images/" + image_id + ".jpg")

    # image_width = scenegraghs[image_id]["width"]
    # image_height = scenegraghs[image_id]["height"]

    x = scenegraghs[image_id]['objects'][entity_id]['x']
    y = scenegraghs[image_id]['objects'][entity_id]['y']
    w  = scenegraghs[image_id]['objects'][entity_id]['w']
    h = scenegraghs[image_id]['objects'][entity_id]['h']   

    target_object_name = scenegraghs[image_id]['objects'][entity_id]['name']

    image_rectangle = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for object_key, object_value in scenegraghs[image_id]['objects'].items():
        if entity_id != str(object_key) and object_value['name'] == target_object_name:
            x_ = object_value['x']
            y_ = object_value['y']
            w_  = object_value['w']
            h_ = object_value['h']   
            image_rectangle = cv2.rectangle(image_rectangle, (x_, y_), (x_ + w_, y_ + h_), (0, 0, 255), 2)

    print(type(image_rectangle))
    encoded_image = encode_image(image_rectangle, 'jpg')
    b64_src = 'data:image/jpg;base64,{}'.format(encoded_image)
    
    
    
    return render_template("example.html", image_src=b64_src, original_question=original_question, ambiguous_question=ambiguous_question, ambiguous_answer=ambiguous_answer, ambiguous_entity=ambiguous_entity,additional_question=additional_question, additional_answer=additional_answer, entity_id=entity_id, q_id=q_id, image_id=image_id)
    
if __name__ == '__main__' :
    app.run(host = '0.0.0.0', port = 6006, debug = True)

