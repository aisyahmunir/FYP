from datetime import datetime
from flask import Blueprint, jsonify, render_template, request, redirect, session, url_for
from .models import Profile
from .models import Result
from .models import Message
from . import db
from datetime import datetime
from sqlalchemy import func
from sqlalchemy.orm import joinedload
from sqlalchemy.orm.exc import NoResultFound

views = Blueprint('views', __name__)

# Function to fetch notifications
def get_notifications():
    return Message.query.filter_by(cleared=False).all()

# Context processor to add notifications to the context
@views.context_processor
def inject_notifications():
    return dict(notifications=get_notifications())

@views.route('/', methods=['POST', 'GET'])
def index():
    try:
        # Fetch each distinct profile along with its latest result
        profiles = (
            db.session.query(Profile, Result)
            .outerjoin(Result, Result.profile_id == Profile.id)
            .group_by(Profile.id)
            .order_by(Profile.id, Result.date_updated.desc())
            .distinct(Profile.id)
            .all()
        )

        return render_template('index.html', profiles=profiles)
    except NoResultFound as e:
        # Handle the case when no results are found
        profiles = Profile.query.all()
        return render_template('index.html', profiles=[])

@views.route('/profile/show', methods=['POST','GET'])
def display():
    profiles = Profile.query.all()
    return render_template('display.html',profiles=profiles)

@views.route('/profile', methods=['POST','GET'])
def plant_profile():
    if request.method == 'POST':
        date = request.form.get('initial_date')
        date_new = list(date.split("-"))
        y = int(date_new[0])
        m = int(date_new[1])
        d = int(date_new[2])
        name_plant = request.form.get('name_plant')
        type_plant = request.form.get('type_plant')
        device = request.form.get('device')
        new_profile = Profile(initial_date=datetime(y,m,d),name_plant=name_plant, type_plant=type_plant, device=device)
        db.session.add(new_profile)
        db.session.commit()
        print(new_profile)
        return redirect(url_for('views.display'))
    return render_template('profile.html')

@views.route('/profile/modify', methods=['POST','GET'])
def modify():
    profiles = Profile.query.all()
    return render_template('modify.html',profiles=profiles)

@views.route('/profile/modify/delete', methods=['POST','GET'])
def delete_profile():
    if request.method == 'POST':
        choice = request.form['choice']
        profile = Profile.query.filter(Profile.id == choice).first()
        db.session.delete(profile)
        db.session.commit()
        return redirect(url_for('views.modify'))
    return redirect(url_for('views.modify'))

@views.route('/notifications')
def notifications():

    # Retrieve messages with associated Result and Profile using outerjoin
    messages = Message.query \
        .filter_by(cleared=False) \
        .outerjoin(Result, Message.result_id == Result.id) \
        .outerjoin(Profile, Result.profile_id == Profile.id) \
        .options(joinedload(Message.result).joinedload(Result.profile)) \
        .all()
    
    # Fetch uncleared messages with associated Result and Profile
    uncleared_messages = Message.query \
        .filter_by(cleared=False) \
        .outerjoin(Result, Message.result_id == Result.id) \
        .outerjoin(Profile, Result.profile_id == Profile.id) \
        .options(joinedload(Message.result).joinedload(Result.profile)) \
        .all()

    # Fetch cleared messages with associated Result and Profile
    cleared_messages = Message.query \
        .filter_by(cleared=True) \
        .outerjoin(Result, Message.result_id == Result.id) \
        .outerjoin(Profile, Result.profile_id == Profile.id) \
        .options(joinedload(Message.result).joinedload(Result.profile)) \
        .all()

    return render_template('notifications.html', uncleared_messages=uncleared_messages, cleared_messages=cleared_messages)

@views.route('/clear_message/<int:message_id>', methods=['POST'])
def clear_message(message_id):
    message = Message.query.get_or_404(message_id)

    # Update the cleared attribute to True
    message.cleared = True
    db.session.commit()

    # Respond with a JSON indicating success
    return jsonify({'status': 'success'})

@views.route('/delete_notifications/<section>', methods=['POST'])
def delete_notification(section):
     # Update the specified messages with cleared=True based on the section
    if section == 'uncleared':
        messages = Message.query.filter_by(cleared=False).all()
    elif section == 'cleared':
        messages = Message.query.filter_by(cleared=True).all()
    else:
        return jsonify(success=False, error="Invalid section")

    if messages:
        for message in messages:
            db.session.delete(message)

        db.session.commit()

        return jsonify(success=True)
    else:
        return jsonify(success=False, error="No messages found in the specified section")

    
@views.route('/testing')
def test():
    latest_record = Result.query.order_by(Result.id.desc()).first()

    if latest_record:
        # Delete the latest record
        db.session.delete(latest_record)
        db.session.commit()
        print(f"Latest record deleted: {latest_record}")
    else:
        print("No records to delete.")
    
    return render_template('index.html')
