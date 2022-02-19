import model  # Import the python file containing the ML model
from flask import Flask, request, render_template  # Import flask libraries

# Initialize the flask class and specify the templates directory
app = Flask(__name__, template_folder="templates")


# Default route set as 'home'
@app.route('/home')
def home():
    return render_template('home.html')  # Render home.html

@app.route('/eda')
def eda():
    return render_template('eda.html')  # Render home.html


# Route 'classify' accepts GET request
@app.route('/classify', methods=['GET'])
def classify_type():
    try:
        asiste_mantencion = request.args.get('man')  # Get parameters for sepal length
        cliente_financiamiento = request.args.get('fin')  # Get parameters for sepal width
        cliente_seguros = request.args.get('seg')  # Get parameters for petal length
        genero = request.args.get('gen')  # Get parameters for petal width
        gse = request.args.get('gse')  # Get parameters for petal width
        generacion = request.args.get('gne')  # Get parameters for petal width
        segmento_sub = request.args.get('sub')  # Get parameters for petal width

        # Get the output from the classification model
        cluster = model.classify(asiste_mantencion, cliente_financiamiento, cliente_seguros, genero, gse, generacion, segmento_sub)

        # Render the output in new HTML page
        return render_template('output.html', cluster=cluster)
    except ValueError:
        return 'Error'


# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
