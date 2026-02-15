from .algebra_logic.modular_powers import potenze_modulari_bp
from .algebra_logic.hasse_diagram import diagramma_hasse_bp
from .algebra_logic.boolean_polynomials import polinomi_booleani_bp
from .algebra_logic.chinese_remainder_theorem import teorema_cinese_resto_bp
from .algebra_logic.bezout_identity import identita_bezout_bp

from .linear_algebra.laplace_determinant import det_laplace_bp
from .linear_algebra.eigenvalues_eigenvectors import eigen_bp
from .linear_algebra.linear_systems import syslin_bp

from .controls.linearization import linearizzazione_bp
from .controls.spectral_decomposition import spectral_bp
from .controls.differential_equations import equazioni_differenziali_bp
from .controls.difference_equations import equazioni_differenze_bp
from .controls.dynamical_systems import sistemi_dinamici_bp

from .operations_research.simplex import simplesso_bp
from .operations_research.complementary_conditions import cc_bp

from .analysis.function_study import studio_funzione_bp
from .analysis.integrals import integrali_bp

def register_routes(app):
    app.register_blueprint(potenze_modulari_bp)
    app.register_blueprint(diagramma_hasse_bp)
    app.register_blueprint(polinomi_booleani_bp)
    app.register_blueprint(linearizzazione_bp)
    app.register_blueprint(spectral_bp)
    app.register_blueprint(equazioni_differenziali_bp)
    app.register_blueprint(equazioni_differenze_bp)
    app.register_blueprint(sistemi_dinamici_bp)
    app.register_blueprint(teorema_cinese_resto_bp)
    app.register_blueprint(identita_bezout_bp)
    app.register_blueprint(det_laplace_bp)
    app.register_blueprint(eigen_bp)
    app.register_blueprint(syslin_bp)
    app.register_blueprint(simplesso_bp)
    app.register_blueprint(cc_bp)
    app.register_blueprint(studio_funzione_bp)
    app.register_blueprint(integrali_bp)
