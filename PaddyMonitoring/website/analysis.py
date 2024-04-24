# Change file path: Line 44, 48, 125, 359

import os
import time
from flask import Blueprint, flash, jsonify, render_template, request, redirect, session, url_for
import firebase_admin
from firebase_admin import credentials, storage
import datetime
from flask_wtf import FlaskForm
import paho.mqtt.client as paho  # mqtt library
from sqlalchemy import func

import urllib3
from wtforms import SelectField

from . import db
from .models import Profile
from .models import Result
from .models import Message

# For analysis
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import cv2
from collections import Counter
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000 
from PIL import Image  
import re
import tensorflow as tf
import torch
from torchvision import transforms


analysis = Blueprint('analysis', __name__)

cred_image = credentials.Certificate("/Users/adrianasfea/Downloads/Paddy Monitoring/paddy-image-firebase-adminsdk-qeti1-80a9116314.json")
firebase_admin.initialize_app(cred_image, {'storageBucket': 'paddy-image.appspot.com'})

# Load the saved model
# loaded_model = tf.keras.models.load_model("/Users/adrianasfea/Downloads/Paddy Monitoring/model/paddy.h5", compile=False)

class MyForm(FlaskForm):
    profile = SelectField('Select an option', choices=[])
    result_date = SelectField('Select a date', choices=[])
   # profile = SelectField('Select an option', choices=[('', 'Choose a profile')] + [(profile.id, profile.name_plant) for profile in Profile.query.all()])

@analysis.route('/dashboard_option', methods=['GET','POST'])
def dashboard_option():
    form = MyForm()
    form.profile.choices = [('', 'Choose a profile')]+[(profile.id, profile.name_plant) for profile in Profile.query.all()]

    if request.method == 'POST':
        selected_option = Profile.query.get(form.profile.data)
        selected_date = Result.query.get(form.result_date.data) # This would be in String
        return redirect(url_for('analysis.display_dash', profile_id=selected_option.id, date=selected_date.date_updated))
    
    return render_template('option.html',form=form)

# Assuming your datetime strings are in the format "%Y-%m-%d %H:%M:%S.%f"
def format_datetime_for_html(datetime_str):
    from datetime import datetime
    dt_object = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
    return dt_object.strftime('%Y-%m-%d')

@analysis.route('/get_result_dates/<int:profile_id>')
def get_result_dates(profile_id):
    # Query the database to get profile_id and distinct dates for the given profile_id
    results = db.session.query(Result.id, Result.date_updated).filter(Result.profile_id == profile_id).distinct().all()

    # Create a list of dictionaries containing profile_id and date_updated
    result_data = [{"id": result.id, "date_updated": format_datetime_for_html(str(result.date_updated))} for result in results]

    # Return the result date choices as JSON
    return jsonify(result_data)

@analysis.route('/dashboard')
def display_dash():
    profiles = Profile.query.all()
    default_profile = Profile.query.first()
    # Query the profile by profile_id. The profile id is taken from choose dash
    # If no profile id passed, default is first profile id in db
    profile_id = request.args.get('profile_id', default_profile.id if default_profile else None)
    profile = Profile.query.get(profile_id)

    if profile:
        from datetime import datetime, timedelta

        # Query the result for the profile, if date is chosen, display the results from that dat
        # Else show latest result
        date_choice = request.args.get('date', None)

        if date_choice:
            latest_result = Result.query.filter_by(profile_id=profile.id, date_updated=date_choice).first()
        
        else:
            latest_result = Result.query.filter_by(profile_id=profile.id).order_by(Result.date_updated.desc()).first()
        
        health_stat = request.args.get('health', None)
        # print("Health status: " + health_stat)
        # health_stat = "Unhealthy"
        if health_stat == "Unhealthy":
            flash('Disease detected in the plant!', 'danger')

            # Save to the database
            message = Message(text='Disease detected in the plant!', category='danger', result_id=latest_result.id)
            db.session.add(message)
            db.session.commit()

        temp = re.findall(r'\d+', latest_result.rgb_val)
        res = list(map(int, temp))

        x = res[0]
        y = res[1]
        z = res[2]
        
        img = Image.new('RGB',(200,200),(x,y,z))
        image_file = "/Users/adrianasfea/Downloads/Paddy Monitoring/Paddy_Monitoring/website/static/temp_image.png"
        img.save(image_file)

        # Create a download URL
        expiration_time = datetime.utcnow() + timedelta(hours=1)
        download_url = storage.bucket().blob(latest_result.image).generate_signed_url(expiration=expiration_time)

        return render_template('dashboard.html', profiles=profiles, profile=profile, result=latest_result, image_url=download_url)

    return "Profile not found"

@analysis.route('/analyse', methods=['GET','POST'])
def get_image():
    form = MyForm()

    form.profile.choices = [(profile.id, profile.name_plant) for profile in Profile.query.all()]
    
    if request.method == 'POST':

        selected_option = Profile.query.get(form.profile.data)
        plant = selected_option.name_plant
        result = analysis_process(plant)

        stage_num = result[0]
        rgb_val = result[1]
        leaf_l = result[2]
        days_since = result[3]
        filename = result[4]
        current_date = result[5]

        # Disease detection
        disease = result[6]
        health_stat = result[7]

        if stage_num==1:
            Stages = "Vegetative"
        elif stage_num==2:
            Stages = "Reproductive"
        else:
            Stages = "Ripening"
        
        
        # Not done! 
        rgb_str = str(rgb_val)

        new_result = Result(stage=Stages,rgb_val=rgb_str,days_since=days_since,date_updated=current_date,image=filename,
                            leaf_l=leaf_l,disease=disease,health_stat=health_stat,profile_id=selected_option.id)
        db.session.add(new_result)
        db.session.commit()

        # Send the image file to the frontend
        return redirect(url_for('analysis.display_dash', profile_id=selected_option.id, health=health_stat))
        
    return render_template('date.html', form=form)

def analysis_process(plant):

    chosen_plant = Profile.query.filter_by(name_plant=plant).first()
    date = chosen_plant.initial_date
    date_new1 = str(date.strftime("%d-%m-%Y"))
    date_new = list(date_new1.split("-"))
    d = int(date_new[0])
    m = int(date_new[1])
    y = int(date_new[2])
    
    device_name = chosen_plant.device
    # This one is to calculate difference with initial date
    now = datetime.datetime.now()

    # Get todays date (This one is to put inside the filename)
    current_date = str(now.strftime("%d%m%Y"))

    # Calculate days since first day of the plant
    initialDate = datetime.datetime(y,m,d) #Attention: month is zero-based
    difference = now - initialDate
    secondsPerDay = 24 * 60 * 60
    daysSince = math.floor(difference.total_seconds() / secondsPerDay)

    bucket = storage.bucket()
    # blob = bucket.get_blob("Legume1-"+current_date+".jpg")
    filename = device_name+"-"+current_date+".jpg"
    # filename = device_name+"-16062023.jpg"
    blob = bucket.get_blob(filename)
    arr = np.frombuffer(blob.download_as_string(),np.uint8)
    image = cv2.imdecode(arr,cv2.COLOR_BGR2BGR555) #actual image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))

    def RGB_HEX(color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


    def get_colors(image, number_of_colors, show_chart):
        reshaped_image = cv2.resize(image, (600, 400))
        reshaped_image = reshaped_image.reshape(reshaped_image.shape[0]*reshaped_image.shape[1], 3)
        clf = KMeans(n_clusters=number_of_colors, n_init='auto')
        #n_init: number of times the k-means algorithm is run with different centroid seeds (auto means no of runs depends on value of init)
        labels = clf.fit_predict(reshaped_image) #compute cluster centers & predict cluster index for each sample
        counts = Counter(labels)
        counts = dict(sorted(counts.items()))
        center_colors = clf.cluster_centers_
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [RGB_HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]
        if show_chart:
            plt.figure(figsize = (8, 6))
            plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        return rgb_colors

    arrays=get_colors(image, 2, True)


    if arrays[0][0]>=245 and arrays[0][0]<255 and arrays[0][1]>=245 and arrays[0][1]<255 and arrays[0][2]>=245 and arrays[0][2]<255:
        arrays = np.delete(arrays,0,0)
    else:
        arrays = np.delete(arrays,1,0)
    rgb_values = np.concatenate(arrays)


    d=[0]*3 #distance of colors
    rgb1 = np.array(rgb_values)
    # colors = [[255,255,255],[127,156,72],[175,200,73]]
    colors = [[131,140,50],[79,117,30],[183,174,86]]
    Point = [0]*3 #point for comparison

    #rgb values
    color1_rgb = sRGBColor(rgb1[0],rgb1[1],rgb1[2])
    # Convert from RGB to Lab Color Space
    color1_lab = convert_color(color1_rgb, LabColor)

    for i in range(len(d)):
        rgb2 = np.array(colors[i])
        color2_rgb = sRGBColor(rgb2[0],rgb2[1],rgb2[2])
        color2_lab = convert_color(color2_rgb, LabColor)
        # Find the color difference
        d[i] = delta_e_cie2000(color1_lab, color2_lab)

    total_color = sum(d)

    for i in range(len(Point)):
        Point[i] += 100-(d[i]/total_color)*100

        # Function to show array of images (intermediate results)
    def show_images(images):
        for i, img in enumerate(images):
            cv2.imshow("image",img)   
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # Find contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)

    # Remove contours which are not large enough
    cnts = [x for x in cnts if cv2.contourArea(x) > 100]

    # Reference object dimensions
    # Here for reference I have used a 2cm x 2cm square
    ref_object = cnts[0]
    box = cv2.minAreaRect(ref_object)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box
    dist_in_pixel = euclidean(tl, tr)
    # dist_in_cm = 2
    dist_in_cm = 4 #part ni tukar la letak 30 dulu, 
    pixel_per_cm = dist_in_pixel/dist_in_cm

    for cnt in cnts:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
        wid = euclidean(tl, tr)/pixel_per_cm
        ht = euclidean(tr, br)/pixel_per_cm
        cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


    heights = []
    for cnt in cnts:
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        (tl, tr, br, bl) = box
        ht = euclidean(tr, br)/pixel_per_cm
        heights.append(ht)

    heights.sort()
    leaf_length = heights[len(heights)-1]

    # length = [0.0,25.0,45.0] #length for each stage
    length = [30.0,90.0,70.0]
    l = [0]*3

    #compare length
    for i in range(len(l)):
        l[i] = abs(leaf_length - length[i])

    total_length = sum(l)

    for i in range(len(Point)):
        Point[i] += 100-(l[i]/total_length)*100

    Stage = Point.index(max(Point))+1 #paling banyak point

    rgb_values = np.int_(rgb_values)

    x = rgb_values[0]
    y = rgb_values[1]
    z = rgb_values[2]

    final_rgb = [x,y,z]
    final_length = "{:.1f}".format(leaf_length)

    img = Image.new('RGB',(200,200),(x,y,z))
    image_file = "/Users/adrianasfea/Downloads/Paddy Monitoring/Paddy_Monitoring/website/static/temp_image.png"
    img.save(image_file)

    if Stage==1:
        Stages = "Vegetative"
    elif Stage==2:
        Stages = "Reproductive"
    else:
        Stages = "Ripening"

    # Disease detection
    # Load the YOLOv8 model
    # yolo_model = torch.hub.load('yolov8', 'custom', path_or_model='/Users/adrianasfea/Downloads/Paddy Monitoring/model/paddydisease.pt')
    # yolo_model.eval()
    
    # class_names = ['Bacterialblight', 'Blast', 'Brownspot']
    # disease, health_stat = predict_disease(yolo_model, filename, class_names, confidence_threshold=0.7)

    disease = "None"
    health_stat = "Healthy"

    if device_name == "Paddy1":
        ACCESS_TOKEN = '0EEoShlBUK2TGA3pEeqI'  # Token of your device
    elif device_name == "Paddy2":
        ACCESS_TOKEN = 'E6PjL7XtSKcXaYGzO2tO'  # Token of your device
    
    broker = "thingsboard.cloud"  # host name
    port = 1883  # data listening port

    def on_publish(client, userdata, result):  # create function for callback
        print("data published to thingsboard \n")
        pass

    client1 = paho.Client("control1")  # create client object
    client1.on_publish = on_publish  # assign function to callback
    client1.username_pw_set(ACCESS_TOKEN)  # access token from thingsboard device
    client1.connect(broker, port, keepalive=60)  # establish connection

    payload="{"
    payload+="\"Date\":"+str(now.strftime("%Y-%m-%d"))+",";
    payload+="\"Day\":"+str(daysSince)+",";
    payload+="\"Disease Detection\":"+str(disease)+",";
    payload+="\"Health Status\":"+str(health_stat)+",";
    payload+="\"Stage\":"+str(Stages)+",";
    payload+="\"RGB values\":"+str(final_rgb)+",";
    payload+="\"Length in cm\":"+str(final_length);
    payload+="}"
    ret= client1.publish("v1/devices/me/telemetry",payload) #topic- v1/devices/me/telemetry
    print("Please check LATEST TELEMETRY field of your device")
    print(payload);
    time.sleep(5)

    if ret.rc == paho.MQTT_ERR_SUCCESS:
        print("Telemetry data published successfully")
    else:
        print(f"Failed to publish telemetry data. Error code: {ret.rc}")
        
    return [Stage, final_rgb, final_length, daysSince, filename, now, disease, health_stat]

# Function to preprocess and predict disease
def predict_disease(model, filename, class_names, confidence_threshold=0.7):
    from datetime import datetime, timedelta

    # Load the YOLOv8 model
    yolo_model_path = "/Users/adrianasfea/Downloads/Paddy Monitoring/model/paddydisease.pt"
    yolo_model = torch.hub.load('yolov8', 'custom', path_or_model=yolo_model_path)
    yolo_model.eval()

    # Create a download URL
    expiration_time = datetime.utcnow() + timedelta(hours=1)
    image_url = storage.bucket().blob(filename).generate_signed_url(expiration=expiration_time)
    image_url = tf.keras.utils.get_file('Plant', origin=image_url )

    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_url, target_size=(256, 256))
    os.remove(image_url)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Normalize the pixel values
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class and confidence
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(100 * predictions[0][predicted_index], 2)

    # Check confidence against the threshold
    if confidence < confidence_threshold:
        predicted_class = "None"
        confidence = 0.0
        health_stat = "Healthy"
    else:
        health_stat = "Unhealthy"


    return predicted_class, health_stat
