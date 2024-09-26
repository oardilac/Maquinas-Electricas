import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import interp1d
from scipy.optimize import fsolve


# Clase para almacenar los parámetros de un generador
class Generador:
    def __init__(
        self,
        RA,
        XS,
        Snom,
        Vnom,
        fpnom,
        num_polos,
        curva_mag_IF,
        curva_mag_EA,
        fsc,
        IF_oper,
        Pnuc,
        Pfyr,
        Pmisc,
        Pmotor,
    ):
        self.RA = RA  # Resistencia de armadura (Ohm)
        self.XS = XS  # Reactancia síncrona (Ohm)
        self.Snom = Snom  # Potencia nominal (VA)
        self.Vnom = Vnom  # Tensión nominal (V)
        self.fpnom = fpnom  # Factor de potencia nominal
        self.num_polos = num_polos  # Número de polos
        self.curva_mag_IF = curva_mag_IF  # Corriente de campo (A)
        self.curva_mag_EA = curva_mag_EA  # FEM interna (V)
        self.fsc = fsc  # Frecuencia sin carga (Hz)
        self.IF_oper = IF_oper  # Corriente de campo en operación (A)
        self.Pnuc = Pnuc  # Pérdidas en el núcleo (W)
        self.Pfyr = Pfyr  # Pérdidas por fricción y ventilación (W)
        self.Pmisc = Pmisc  # Pérdidas misceláneas (W)
        self.Pmotor = Pmotor  # Capacidad del motor primario (W)

    def interpolar_EA(self, IF):
        """
        Interpola la FEM interna (EA) para una dada corriente de campo (IF) usando la curva de magnetización.
        """
        f = interp1d(
            self.curva_mag_IF, self.curva_mag_EA, kind="cubic", fill_value="extrapolate"
        )
        return f(IF)


class Carga:
    def __init__(self, R_load, X_load):
        self.R_load = R_load
        self.X_load = X_load


def calcular_parametros(gen1, gen2, carga):
    """
    Calcula los parámetros eléctricos y mecánicos de los generadores en paralelo alimentando una carga.
    Utiliza el método de Newton-Raphson para resolver el sistema de ecuaciones no lineales.
    """

    # Función que define el sistema de ecuaciones
    def sistema_ecuaciones(vars):
        VT, IA1_re, IA1_im, IA2_re, IA2_im, delta = vars  # Variables a resolver

        # Tensión terminal en forma de phasor (ángulo 0)
        VT_complex = VT  # VT ∠ 0°

        # Corrientes de armadura en phasor
        IA1 = IA1_re + 1j * IA1_im
        IA2 = IA2_re + 1j * IA2_im

        # Corriente de carga calculada correctamente
        Z_load = carga.R_load + 1j * carga.X_load
        IL = VT_complex / Z_load
        IL_re = IL.real
        IL_im = IL.imag

        # FEM internas mediante la interpolación de las curvas de magnetización
        EA1 = gen1.interpolar_EA(gen1.IF_oper)  # IF1 = IF_oper1
        EA2 = gen2.interpolar_EA(gen2.IF_oper)  # IF2 = IF_oper2

        # Ecuaciones para el Generador 1
        # EA1 ∠ delta = VT + IA1 * (RA1 + j XS1)
        EA1_phasor = EA1 * np.exp(1j * delta)
        ecuacion1 = np.real(EA1_phasor) - (VT + IA1_re * gen1.RA - IA1_im * gen1.XS)
        ecuacion2 = np.imag(EA1_phasor) - (IA1_re * gen1.XS + IA1_im * gen1.RA)

        # Ecuaciones para el Generador 2
        # EA2 ∠ delta = VT + IA2 * (RA2 + j XS2)
        EA2_phasor = EA2 * np.exp(1j * delta)
        ecuacion3 = np.real(EA2_phasor) - (VT + IA2_re * gen2.RA - IA2_im * gen2.XS)
        ecuacion4 = np.imag(EA2_phasor) - (IA2_re * gen2.XS + IA2_im * gen2.RA)

        # Balance de corrientes
        # IA1 + IA2 = IL
        ecuacion5 = IA1_re + IA2_re - IL_re
        ecuacion6 = IA1_im + IA2_im - IL_im

        return [ecuacion1, ecuacion2, ecuacion3, ecuacion4, ecuacion5, ecuacion6]

    # Valores iniciales para la solución (VT, IA1_re, IA1_im, IA2_re, IA2_im, delta)
    VT_init = gen1.Vnom  # Tensión nominal como valor inicial
    IA_init = gen1.Snom / gen1.Vnom  # Corriente nominal como valor inicial
    delta_init = 0.1  # Pequeño ángulo inicial en radianes

    vars_iniciales = [VT_init, IA_init, 0.0, IA_init, 0.0, delta_init]

    # Resolver el sistema de ecuaciones usando fsolve
    solucion, info, ier, mesg = fsolve(
        sistema_ecuaciones, vars_iniciales, full_output=True
    )

    if ier != 1:
        raise ValueError(f"No se pudo converger: {mesg}")

    # Extraer las variables solucionadas
    VT_sol, IA1_re_sol, IA1_im_sol, IA2_re_sol, IA2_im_sol, delta_sol = solucion

    # Calcular IL (se puede recalcular para verificar)
    IL = (IA1_re_sol + IA2_re_sol) + 1j * (IA1_im_sol + IA2_im_sol)

    # Calcular Potencias
    # Potencia por generador: S = VT * IA * conjugado
    IA1 = IA1_re_sol + 1j * IA1_im_sol
    IA2 = IA2_re_sol + 1j * IA2_im_sol
    IL = IA1 + IA2

    S1 = VT_sol * np.conj(IA1)
    S2 = VT_sol * np.conj(IA2)
    SL = VT_sol * np.conj(IL)

    # Calcular ángulo de par (delta)
    # Ya está calculado como delta_sol

    # Calcular Torques
    # Torque inducido: Tind = P / omega_sinc
    # Velocidad síncrona: omega_sinc = 2 * pi * fe / p
    omega_sinc = 2 * np.pi * gen1.fsc / gen1.num_polos

    P1 = np.real(S1)
    Q1 = np.imag(S1)
    P2 = np.real(S2)
    Q2 = np.imag(S2)
    PL = np.real(SL)
    QL = np.imag(SL)

    Tind1 = P1 / omega_sinc
    Tind2 = P2 / omega_sinc
    Tind_total = Tind1 + Tind2

    # Torque aplicado por los motores primarios
    Tap = (gen1.Pmotor + gen2.Pmotor) / omega_sinc  # Suma de capacidades

    # Frecuencia eléctrica
    fe = gen1.fsc  # Frecuencia sin carga ajustada

    # Pérdidas en cobre
    PCu1 = (IA1_re_sol**2 + IA1_im_sol**2) * gen1.RA
    PCu2 = (IA2_re_sol**2 + IA2_im_sol**2) * gen2.RA
    PCu_total = PCu1 + PCu2

    # Compilar resultados
    resultados = {
        "VT (V)": VT_sol,
        "IA1 (A)": np.abs(IA1),
        "IA2 (A)": np.abs(IA2),
        "IL (A)": np.abs(IL),
        "IF1 (A)": gen1.IF_oper,  # Simplificación
        "IF2 (A)": gen2.IF_oper,  # Simplificación
        "P1 (W)": P1,
        "Q1 (VAR)": Q1,
        "S1 (VA)": np.abs(S1),
        "P2 (W)": P2,
        "Q2 (VAR)": Q2,
        "S2 (VA)": np.abs(S2),
        "PL (W)": PL,
        "QL (VAR)": QL,
        "SL (VA)": np.abs(SL),
        "delta (rad)": delta_sol,
        "Tind1 (Nm)": Tind1,
        "Tind2 (Nm)": Tind2,
        "Tind_total (Nm)": Tind_total,
        "Tap (Nm)": Tap,
        "omega_sinc (rad/s)": omega_sinc,
        "fe (Hz)": fe,
        "PCu1 (W)": PCu1,
        "PCu2 (W)": PCu2,
        "PCu_total (W)": PCu_total,
    }

    return resultados


def graficar_curvas_mag(gen1, gen2, resultados):
    """
    Grafica las curvas de magnetización de ambos generadores y muestra el punto de operación.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Curva de Magnetización Generador 1
    ax.plot(gen1.curva_mag_IF, gen1.curva_mag_EA, label="Generador 1")
    # Punto de Operación Generador 1
    EA1_op = gen1.interpolar_EA(gen1.IF_oper)
    ax.plot(gen1.IF_oper, EA1_op, "ro", label="Operación G1")

    # Curva de Magnetización Generador 2
    ax.plot(gen2.curva_mag_IF, gen2.curva_mag_EA, label="Generador 2")
    # Punto de Operación Generador 2
    EA2_op = gen2.interpolar_EA(gen2.IF_oper)
    ax.plot(gen2.IF_oper, EA2_op, "go", label="Operación G2")

    ax.set_title("Curvas de Magnetización")
    ax.set_xlabel("Corriente de Campo IF (A)")
    ax.set_ylabel("FEM Interna EA (V)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def graficar_curvas_capacidad(gen1, gen2, resultados):
    """
    Grafica las curvas de capacidad de ambos generadores y muestra el punto de operación.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Curva de Capacidad Generador 1
    S_max1 = gen1.Snom
    Q1 = np.linspace(-S_max1, S_max1, 100)
    P1 = np.sqrt(S_max1**2 - Q1**2)
    ax.plot(P1, Q1, label="Generador 1")
    # Punto de Operación Generador 1
    ax.plot(resultados["P1 (W)"], resultados["Q1 (VAR)"], "ro", label="Operación G1")

    # Curva de Capacidad Generador 2
    S_max2 = gen2.Snom
    Q2 = np.linspace(-S_max2, S_max2, 100)
    P2 = np.sqrt(S_max2**2 - Q2**2)
    ax.plot(P2, Q2, label="Generador 2")
    # Punto de Operación Generador 2
    ax.plot(resultados["P2 (W)"], resultados["Q2 (VAR)"], "go", label="Operación G2")

    ax.set_title("Curvas de Capacidad")
    ax.set_xlabel("Potencia Activa P (W)")
    ax.set_ylabel("Potencia Reactiva Q (VAR)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


def main():
    st.title("Análisis de Generadores Síncronos en Paralelo")

    st.sidebar.header("Parámetros de los Generadores")

    # Entrada de parámetros para el Generador 1
    st.sidebar.subheader("Generador 1")
    RA1 = st.sidebar.number_input(
        "Resistencia de Armadura RA1 (Ω)", value=0.01, format="%.5f"
    )
    XS1 = st.sidebar.number_input(
        "Reactancia Síncrona XS1 (Ω)", value=1.0, format="%.5f"
    )
    Snom1 = st.sidebar.number_input(
        "Potencia Nominal Snom1 (VA)", value=100000.0, format="%.2f"
    )
    Vnom1 = st.sidebar.number_input(
        "Tensión Nominal Vnom1 (V)", value=400.0, format="%.2f"
    )
    fpnom1 = st.sidebar.number_input(
        "Factor de Potencia Nominal fpnom1", value=0.85, format="%.2f"
    )
    num_polos1 = st.sidebar.number_input(
        "Número de Polos Generador 1", min_value=2, step=2, value=4
    )
    fsc1 = st.sidebar.number_input(
        "Frecuencia sin carga fsc1 (Hz)", value=60.0, format="%.2f"
    )
    IF_oper1 = st.sidebar.number_input(
        "Corriente de Campo en Operación IF1 (A)", value=25.0, format="%.2f"
    )
    Pnuc1 = st.sidebar.number_input(
        "Pérdidas en el Núcleo Pnuc1 (W)", value=100.0, format="%.2f"
    )
    Pfyr1 = st.sidebar.number_input(
        "Pérdidas por Fricción y Ventilación Pfyr1 (W)", value=50.0, format="%.2f"
    )
    Pmisc1 = st.sidebar.number_input(
        "Pérdidas Misceláneas Pmisc1 (W)", value=20.0, format="%.2f"
    )
    Pmotor1 = st.sidebar.number_input(
        "Capacidad del Motor Primario Pmotor1 (W)", value=500.0, format="%.2f"
    )

    # Entrada de parámetros para el Generador 2
    st.sidebar.subheader("Generador 2")
    RA2 = st.sidebar.number_input(
        "Resistencia de Armadura RA2 (Ω)", value=0.01, format="%.5f"
    )
    XS2 = st.sidebar.number_input(
        "Reactancia Síncrona XS2 (Ω)", value=1.0, format="%.5f"
    )
    Snom2 = st.sidebar.number_input(
        "Potencia Nominal Snom2 (VA)", value=100000.0, format="%.2f"
    )
    Vnom2 = st.sidebar.number_input(
        "Tensión Nominal Vnom2 (V)", value=400.0, format="%.2f"
    )
    fpnom2 = st.sidebar.number_input(
        "Factor de Potencia Nominal fpnom2", value=0.85, format="%.2f"
    )
    num_polos2 = st.sidebar.number_input(
        "Número de Polos Generador 2", min_value=2, step=2, value=4
    )
    curva_mag_IF1_list = [0, 10, 20, 30, 40, 50]
    curva_mag_EA1_list = [0, 100, 200, 300, 400, 500]
    fsc2 = st.sidebar.number_input(
        "Frecuencia sin carga fsc2 (Hz)", value=60.0, format="%.2f"
    )
    IF_oper2 = st.sidebar.number_input(
        "Corriente de Campo en Operación IF2 (A)", value=25.0, format="%.2f"
    )
    Pnuc2 = st.sidebar.number_input(
        "Pérdidas en el Núcleo Pnuc2 (W)", value=100.0, format="%.2f"
    )
    Pfyr2 = st.sidebar.number_input(
        "Pérdidas por Fricción y Ventilación Pfyr2 (W)", value=50.0, format="%.2f"
    )
    Pmisc2 = st.sidebar.number_input(
        "Pérdidas Misceláneas Pmisc2 (W)", value=20.0, format="%.2f"
    )
    Pmotor2 = st.sidebar.number_input(
        "Capacidad del Motor Primario Pmotor2 (W)", value=500.0, format="%.2f"
    )

    st.sidebar.header("Parámetros de la Carga")
    R_load = st.sidebar.number_input(
        "Resistencia de Carga R_load (Ω)", value=50.0, format="%.2f"
    )
    X_load = st.sidebar.number_input(
        "Reactancia de Carga X_load (Ω)", value=30.0, format="%.2f"
    )

    # Crear objetos Generador y Carga
    try:
        gen1 = Generador(
            RA1,
            XS1,
            Snom1,
            Vnom1,
            fpnom1,
            num_polos1,
            curva_mag_IF1_list,
            curva_mag_EA1_list,
            fsc1,
            IF_oper1,
            Pnuc1,
            Pfyr1,
            Pmisc1,
            Pmotor1,
        )
        gen2 = Generador(
            RA2,
            XS2,
            Snom2,
            Vnom2,
            fpnom2,
            num_polos2,
            curva_mag_IF1_list,
            curva_mag_EA1_list,
            fsc2,
            IF_oper2,
            Pnuc2,
            Pfyr2,
            Pmisc2,
            Pmotor2,
        )

        carga = Carga(R_load, X_load)
    except Exception as e:
        st.error(f"Error en la entrada de datos: {e}")
        st.stop()

    # Botón para calcular
    if st.button("Calcular"):
        try:
            resultados = calcular_parametros(gen1, gen2, carga)

            # Mostrar Resultados
            st.header("Resultados de Cálculos")
            for clave, valor in resultados.items():
                if "rad" in clave:
                    st.write(f"**{clave}**: {valor:.4f} rad")
                elif (
                    "Hz" in clave
                    or "V" in clave
                    or "A" in clave
                    or "W" in clave
                    or "VAR" in clave
                    or "VA" in clave
                    or "Nm" in clave
                ):
                    st.write(f"**{clave}**: {valor:.2f}")
                else:
                    st.write(f"**{clave}**: {valor}")

            # Graficar Curvas de Magnetización
            st.header("Curvas de Magnetización")
            graficar_curvas_mag(gen1, gen2, resultados)

            # Graficar Curvas de Capacidad
            st.header("Curvas de Capacidad")
            graficar_curvas_capacidad(gen1, gen2, resultados)

        except Exception as e:
            st.error(f"Ocurrió un error durante los cálculos: {e}")


if __name__ == "__main__":
    main()
