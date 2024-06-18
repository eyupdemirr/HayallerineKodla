from flask import Flask, render_template
from anaSayfa.routes import anaSayfa_bp
from hangiAlanAI.routes import hangiAlanAI_bp
from hangiDilAI.routes import hangiDilAI_bp

app = Flask(__name__, template_folder='anaSayfa/templates', static_folder='anaSayfa/static')
app.secret_key = 'supersecretkey'

# Blueprint'leri kaydet
app.register_blueprint(anaSayfa_bp)

#Blueprint of hangiAlanAI 
app.register_blueprint(hangiDilAI_bp, url_prefix='/hangiDilAI')

#Blueprint of hangiDilAI 
app.register_blueprint(hangiAlanAI_bp, url_prefix='/hangiAlanAI')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
