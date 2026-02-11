from flask import Flask, render_template
from routes import register_routes
import os

app = Flask(__name__)
register_routes(app)

# Homepage route
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/linearizzazione.html")
def linearizzazione():
    return render_template("controlli/linearizzazione.html")

@app.route("/decomposizione-spettrale.html")
def decomposizione():
    return render_template("controlli/decomposizione-spettrale.html")

@app.route("/equazioni-differenziali.html")
def equazioni_differenziali():
    return render_template("controlli/equazioni-differenziali.html")

@app.route("/equazioni-differenze.html")
def equazioni_differenze():
    return render_template("controlli/equazioni-differenze.html")


@app.route("/potenze-modulari.html")
def potenze_modulari():
    return render_template("algebra_logica/potenze-modulari.html")

@app.route("/diagramma-hasse.html")
def diagramma_hasse():
    return render_template("algebra_logica/diagramma-hasse.html")

@app.route("/polinomi-booleani.html")
def polinomi_booleani():
    return render_template("algebra_logica/polinomi-booleani.html")

@app.route("/teorema-cinese-resto.html")
def teorema_cinese_resto():
    return render_template("algebra_logica/teorema-cinese-resto.html")

@app.route("/identita-bezout.html")
def identita_bezout():
    return render_template("algebra_logica/identita-bezout.html")



@app.route("/determinante-laplace.html")
def determinante_laplace():
    return render_template("algebra_lineare/determinante-laplace.html")

@app.route("/autovalori-autovettori.html")
def autovalori_autovettori():
    return render_template("algebra_lineare/autovalori-autovettori.html")

@app.route("/sistemi-lineari.html")
def sistemi_lineari():
    return render_template("algebra_lineare/sistemi-lineari.html")


@app.route("/simplesso.html")
def simplesso():
    return render_template("ricerca_operativa/simplesso.html")

@app.route("/condizioni-complementari.html")
def condizioni_complementari():
    return render_template("ricerca_operativa/condizioni-complementari.html")

@app.route("/studio-funzione.html")
def studio_funzione():
    return render_template("analisi/studio-funzione.html")

@app.route("/integrali.html")
def integrali():
    return render_template("analisi/integrali.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
