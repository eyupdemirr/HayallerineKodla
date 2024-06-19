from flask import Blueprint, render_template, request, redirect, url_for, flash
import json
import os
from datetime import datetime

anaSayfa_bp = Blueprint('anaSayfa', __name__, template_folder='templates', static_folder='static')
data_file = 'users.json'

# Kullanıcı verilerini saklamak için dosya adı
def load_users():
    if os.path.exists(data_file):
        with open(data_file, 'r') as file:
            return json.load(file)
    return {}

# Kullanıcı verilerini dosyaya yaz
def save_users(users):
    with open(data_file, 'w') as file:
        json.dump(users, file, indent=4)

@anaSayfa_bp.route('/')
def index():
    return render_template('index.html')

@anaSayfa_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = load_users()
        
        if username in users and users[username]['password'] == password:
            flash(f"Hoşgeldiniz, {users[username]['name']} {users[username]['surname']}", "success")
            return f"""
                <script>
                    window.alert("Sayın {username} Hoşgeldiniz.."); window.location.href = "{url_for('anaSayfa.index')}";
                </script>"""
        else:
            flash("Kullanıcı adı veya şifre hatalı.", "danger")
            return redirect(url_for('anaSayfa.login_alert', message="Kullanıcı adı veya şifre hatalı."))
    
    return render_template('login.html')

@anaSayfa_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        name = request.form['name']
        surname = request.form['surname']
        password = request.form['password']
        birthdate = request.form['birthdate']
        
        users = load_users()
        
        if username in users:
            flash("Bu kullanıcı adı zaten alınmış.", "danger")
            return redirect(url_for('anaSayfa.signup_alert', message="Bu kullanıcı adı zaten alınmış."))
        
        try:
            datetime.strptime(birthdate, "%Y-%m-%d")
        except ValueError:
            flash("Geçersiz doğum tarihi formatı. Lütfen YYYY-MM-DD formatında ve gerçekçi tarih bilgileri giriniz.", "danger")
            return redirect(url_for('anaSayfa.signup_alert', message="Geçersiz doğum tarihi formatı. Lütfen YYYY-MM-DD formatında ve gerçekçi tarih bilgileri giriniz."))
        
        users[username] = {
            'name': name,
            'surname': surname,
            'password': password,
            'birthdate': birthdate
        }
        
        save_users(users)
        flash("Kayıt başarılı. Giriş yapabilirsiniz.", "success")
        return f"""
            <script>
                window.alert("Sayın {username} Hoşgeldiniz..");
                window.location.href = "{url_for('anaSayfa.index')}";
            </script>
            {redirect(url_for('anaSayfa.login'))}
            """
    
    return render_template('signup.html')

@anaSayfa_bp.route('/login_alert')
def login_alert():
    message = request.args.get('message', '')
    return f"""
    <script>window.alert("{message}"); window.location.href = "{url_for('anaSayfa.login')}";</script>
    """

@anaSayfa_bp.route('/signup_alert')
def signup_alert():
    message = request.args.get('message', '')
    return f"""
    <script>window.alert("{message}"); window.location.href = "{url_for('anaSayfa.signup')}";</script>
    """
