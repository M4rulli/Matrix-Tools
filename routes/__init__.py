from .algebra_logica.potenze_modulari import potenze_modulari_bp
from .algebra_logica.diagramma_hasse import diagramma_hasse_bp
from .algebra_logica.polinomi_booleani import polinomi_booleani_bp
from .algebra_logica.teorema_cinese_resto import teorema_cinese_resto_bp
from .algebra_logica.identita_bezout import identita_bezout_bp

from .algerba_lineare.determinante_laplace import det_laplace_bp
from .algerba_lineare.autovalori_autovettori import eigen_bp
from .algerba_lineare.sistemi_lineari import syslin_bp

from .controlli.linearizzazione import linearizzazione_bp
from .controlli.decomposizione_spettrale import spectral_bp
from .controlli.equazioni_differenziali import equazioni_differenziali_bp
from .controlli.equazioni_differenze import equazioni_differenze_bp

from .ricerca_operativa.simplesso import simplesso_bp
from .ricerca_operativa.condizioni_complementari import cc_bp

from .analisi.studio_funzione import studio_funzione_bp
from .analisi.integrali import integrali_bp

def register_routes(app):
    app.register_blueprint(potenze_modulari_bp)
    app.register_blueprint(diagramma_hasse_bp)
    app.register_blueprint(polinomi_booleani_bp)
    app.register_blueprint(linearizzazione_bp)
    app.register_blueprint(spectral_bp)
    app.register_blueprint(equazioni_differenziali_bp)
    app.register_blueprint(equazioni_differenze_bp)
    app.register_blueprint(teorema_cinese_resto_bp)
    app.register_blueprint(identita_bezout_bp)
    app.register_blueprint(det_laplace_bp)
    app.register_blueprint(eigen_bp)
    app.register_blueprint(syslin_bp)
    app.register_blueprint(simplesso_bp)
    app.register_blueprint(cc_bp)
    app.register_blueprint(studio_funzione_bp)
    app.register_blueprint(integrali_bp)
