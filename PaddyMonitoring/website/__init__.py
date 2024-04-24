# Def create app and create database

from datetime import datetime
from flask import Flask
from os import path
from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()
DB_NAME = "database.db"

def create_app():
    app = Flask(__name__, static_folder='static')
    app.config['SECRET_KEY'] = 'hjshjhdjah kjshkjdhjs'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # New addition
    app.config['SESSION_PERMANENT'] = True  # Session data will be stored permanently
    db.init_app(app)
    # app.config['STATIC_URL_PATH'] = '/website/static'

    from .views import views
    from .analysis import analysis

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(analysis, url_prefix='/')

    # Assets Management
    app.config['ASSETS_ROOT'] = '/static/assets'
    # ASSETS_ROOT = os.getenv('ASSETS_ROOT', '/static/assets') 

    IMAGE_FOLDER = os.path.join('static', 'images')
    app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER 

    # Add the custom filter to the Jinja2 environment
    app.jinja_env.filters['format_datetime'] = format_datetime
    app.jinja_env.filters['format_date'] = format_date
    
    with app.app_context():
        db.create_all()

    return app

def create_database(app):
    if not path.exists('website/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')

def format_datetime(value):
    datetime_object = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
    formatted_datetime = datetime_object.strftime("%d-%m-%Y %H:%M:%S")
    return formatted_datetime

def format_date(value):
    datetime_object = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f")
    formatted_datetime = datetime_object.strftime("%d-%m-%Y")
    return formatted_datetime