from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
# def hello_world():  # put application's code here
#     return 'Hello World!'
def home_screen():
    return render_template('home.html')


@app.route('/generate_caption.html')
# def hello_world():  # put application's code here
#     return 'Hello World!'
def generate_caption_page():
    return render_template('generate_caption.html')


if __name__ == '__main__':
    app.run()
