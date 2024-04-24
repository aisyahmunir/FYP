# Define database models

from . import db

class Profile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    initial_date = db.Column(db.DateTime)
    name_plant = db.Column(db.String(120))
    type_plant = db.Column(db.String(120))
    device = db.Column(db.String(120))
    # Create relationship
    result = db.relationship('Result', backref='profile', lazy=True, cascade='all, delete-orphan')

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stage = db.Column(db.String(120)) # Need to add the string, not the stage number
    rgb_val = db.Column(db.String(120)) # This is list but supposedly list of int can add as str
    days_since = db.Column(db.Integer)
    date_updated = db.Column(db.String(120))
    image = db.Column(db.String(12000))
    # Added new attributes
    leaf_l = db.Column(db.Integer)
    disease = db.Column(db.String(120))
    health_stat = db.Column(db.String(120))
    # Need a foreign key to associate the result with the profile
    profile_id = db.Column(db.Integer, db.ForeignKey('profile.id', ondelete='CASCADE'), nullable=False)
    # Newly added relationship.
    message = db.relationship('Message', backref='result', lazy=True, cascade='all, delete-orphan')

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(255))
    category = db.Column(db.String(120))
    # Newly added for relationship.
    result_id = db.Column(db.Integer, db.ForeignKey('result.id', ondelete='CASCADE'), nullable=False)
    cleared = db.Column(db.Boolean, default=False)

