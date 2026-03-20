"""
solver.py
=========
ORC cycle solver with iterative convergence for recuperator-economiser coupling.

Cycle topology:
  State 1  Turbine inlet           High P, superheated    <- Economiser cold-out
  State 2  Turbine exhaust         Low P,  superheated    -> Recuperator hot-in
  State 3  Recuperator hot-out     Low P,  superheated    -> Condenser inlet
  State 5  Condenser outlet        Low P,  subcooled      -> Pump inlet
  State 6  Pump outlet             High P, liquid          -> Recuperator cold-in
  State 7  Recuperator cold-out    High P, superheated    -> Economiser cold-in
  (1)      Economiser cold-out     High P, superheated    -> Turbine inlet (closes loop)

  Note: State 4 intentionally skipped.

External hot gas (economiser hot side):
  Gas_in -> Economiser -> Gas_out
  T_gas_in, T_gas_out, mdot_gas, Cp_gas all given as inputs.
  Q_econ fixed from gas side; h1 = h7 + Q_econ/mdot_wf is an OUTPUT.

Circular dependency  h1 <-> s1 <-> h2 <-> Q_recup <-> h7 <-> h1
resolved by fixed-point iteration with relaxation.

Key constraint checks:
  - T5  < T_sat(P_cond) < T3     feasible pressure band
  - T7  > T_sat(P_high)           state 7 must be superheated
  - T7  < T_gas_out               cold-end pinch (by construction via approach)
  - T1  < T_gas_in                hot-end pinch
  - T2  > T_sat(P_cond)           turbine exhaust superheated
  - T3  > T_sat(P_cond)           recuperator hot-out superheated
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import thermodynamics as td
import components as comp

logger = logging.getLogger(__name__)


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class CycleConfiguration:
    """
    All fixed inputs for one ORC cycle evaluation.
    Edit main.py to change these -- do not modify here.
    """
    # Working fluid
    fluid: str

    # Pressures [Pa]
    P_high: float                      # turbine inlet pressure

    # Cooling water
    T_cw_in:  float                    # K   cooling water inlet
    dT_cw:    float                    # K   cooling water temperature rise

    # Condenser constraint temperatures [K]
    T3_min:   float                    # minimum condenser inlet (recup hot-out)
    T5:       float                    # condenser outlet (subcooled)

    # External hot gas economiser [all given]
    T_gas_in:  float                   # K
    T_gas_out: float                   # K
    mdot_gas:  float                   # kg/s
    Cp_gas:    float                   # J/kgK

    # Recuperator approach [K]
    # T7 = T3_min + approach_recup  -- sets recuperator cold-out temperature
    approach_recup: float

    # Economiser approach [K]
    # T_gas_out_eff = T7 + approach_econ  -- cold-end pinch
    # Here T_gas_out is given directly so approach_econ is computed, not set.
    # Left here for reference / future use.
    approach_econ: float = 0.0         # K   informational only when T_gas_out is given

    # Component efficiencies
    eta_turbine: float = 0.85
    eta_pump:    float = 0.75

    # Mass flow
    mdot: float = 1.0                  # kg/s working fluid

    # Numerical solver settings
    max_iterations:        int   = 100
    convergence_tol:       float = 1e-4    # J/kg  on h1 residual
    relaxation_factor:     float = 0.5     # 0 < alpha <= 1

    def validate(self) -> None:
        """Validate configuration before solving."""
        if self.P_high <= 0:
            raise ValueError(f"P_high must be positive, got {self.P_high}")
        if self.T3_min <= self.T5:
            raise ValueError(
                f"T3_min={self.T3_min-273.15:.1f}C must be > T5={self.T5-273.15:.1f}C"
            )
        if self.T_gas_in <= self.T_gas_out:
            raise ValueError("T_gas_in must be greater than T_gas_out")
        if not 0 < self.eta_turbine <= 1:
            raise ValueError(f"eta_turbine must be in (0,1], got {self.eta_turbine}")
        if not 0 < self.eta_pump <= 1:
            raise ValueError(f"eta_pump must be in (0,1], got {self.eta_pump}")
        if not 0 < self.relaxation_factor <= 1:
            raise ValueError(f"relaxation_factor must be in (0,1], got {self.relaxation_factor}")
        if self.mdot <= 0:
            raise ValueError(f"mdot must be positive, got {self.mdot}")
        if self.approach_recup <= 0:
            raise ValueError(f"approach_recup must be positive, got {self.approach_recup}")


@dataclass
class StatePoint:
    """Single thermodynamic state point."""
    label:      str
    T:          float          # K
    P:          float          # Pa
    h:          float          # J/kg
    s:          float          # J/kgK
    description: str = ""

    @property
    def T_C(self) -> float:
        return self.T - 273.15

    @property
    def P_bar(self) -> float:
        return self.P / 1e5

    def to_dict(self) -> Dict:
        return {
            "label":       self.label,
            "T_K":         self.T,
            "T_C":         self.T_C,
            "P_Pa":        self.P,
            "P_bar":       self.P_bar,
            "h_Jkg":       self.h,
            "s_JkgK":      self.s,
            "description": self.description,
        }


@dataclass
class CycleResults:
    """
    Complete ORC cycle solution results.
    All energies in J/kg (specific) or W (total power).
    """
    # State points keyed by state number
    states: Dict[int, StatePoint]

    # Specific energies [J/kg]
    w_turbine_specific:  float
    w_pump_specific:     float
    w_net_specific:      float
    q_in_specific:       float       # economiser duty = external heat input
    q_cond_specific:     float       # condenser heat rejection
    Q_recup_specific:    float       # recuperator internal heat transfer

    # Total power [W]
    W_turbine:           float
    W_pump:              float
    W_net:               float

    # Heat duties [W]
    Q_econ_total:        float       # economiser total duty
    Q_cond_total:        float       # condenser total duty

    # Cycle performance
    thermal_efficiency:  float       # w_net / q_in

    # Cooling water
    mdot_cw:             float       # kg/s

    # Economiser gas side
    T_gas_in:            float       # K
    T_gas_out:           float       # K
    pinch_cold_end:      float       # K  T_gas_out - T7
    pinch_hot_end:       float       # K  T_gas_in  - T1

    # Convergence
    converged:           bool
    iterations:          int
    residual_history:    List[float]

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items() if k != "states"}
        d["states"] = {k: v.to_dict() for k, v in self.states.items()}
        return d


# =============================================================================
# SOLVER CLASS
# =============================================================================

class ORCCycleSolver:
    """
    ORC cycle solver with recuperator-economiser iterative convergence.

    Cycle:
      1(Turb in) -> Turbine -> 2(Turb out)
                -> Recup hot -> 3(Recup hot-out)
                -> Condenser  -> 5(Cond out)
                -> Pump        -> 6(Pump out)
                -> Recup cold  -> 7(Recup cold-out)
                -> Economiser  -> 1  (closes loop)

    Circular dependency resolved by fixed-point iteration on h1.
    """

    def __init__(self, config: CycleConfiguration) -> None:
        """
        Initialise solver with a validated CycleConfiguration.

        Parameters
        ----------
        config : CycleConfiguration
        """
        config.validate()
        self.config = config
        self.fluid  = config.fluid
        logger.info(f"ORCCycleSolver initialised for fluid: {self.fluid}")

    # -------------------------------------------------------------------------
    def solve(self, P_cond: float) -> Optional[CycleResults]:
        """
        Solve the complete ORC cycle at condensing pressure P_cond [Pa].

        Parameters
        ----------
        P_cond : float   condensing pressure [Pa]  -- the free variable

        Returns
        -------
        CycleResults if converged and feasible, None if infeasible.
        """
        cfg    = self.config
        fluid  = self.fluid
        P_high = cfg.P_high

        logger.debug(f"Solving at P_cond={P_cond/1e5:.4f} bar")

        # ------------------------------------------------------------------
        # SATURATION CHECK
        # T5 < T_sat(P_cond) < T3_min  -- feasible pressure band
        # ------------------------------------------------------------------
        T_sat     = td.sat_temperature(P_cond, fluid)
        T_sat_high = td.sat_temperature(P_high, fluid)

        if cfg.T5 >= T_sat:
            logger.debug("Infeasible: T5 >= T_sat -- condenser outlet not subcooled")
            return None
        if cfg.T3_min <= T_sat:
            logger.debug("Infeasible: T3_min <= T_sat -- condenser inlet not superheated")
            return None

        # ------------------------------------------------------------------
        # STATE 5  Condenser outlet -- fixed by constraint
        # ------------------------------------------------------------------
        h5 = td.enthalpy(cfg.T5, P_cond, fluid)
        s5 = td.entropy( cfg.T5, P_cond, fluid)
        logger.debug(f"State 5: T={cfg.T5-273.15:.2f}C  h={h5:.1f} J/kg")

        # ------------------------------------------------------------------
        # STATE 6  Pump outlet -- 5 -> 6
        # ------------------------------------------------------------------
        pump_out = comp.pump(h5, s5, P_high, cfg.eta_pump, fluid)
        h6 = pump_out["h"]
        T6 = pump_out["T"]
        logger.debug(f"State 6: T={T6-273.15:.2f}C  h={h6:.1f} J/kg  w_pump={pump_out['w']:.1f} J/kg")

        # ------------------------------------------------------------------
        # STATE 7  Recuperator cold-out -- fixed by approach
        # T7 = T3_min + approach_recup
        # T7 MUST be superheated at P_high.
        # ------------------------------------------------------------------
        T7 = cfg.T3_min + cfg.approach_recup

        if T7 <= T_sat_high:
            logger.debug(
                f"Infeasible: T7={T7-273.15:.2f}C <= T_sat(P_high)={T_sat_high-273.15:.2f}C "
                f"-- state 7 not superheated. Increase approach_recup."
            )
            return None

        h7      = td.enthalpy(T7, P_high, fluid)
        Q_recup = h7 - h6          # J/kg  recuperator cold-side duty

        if Q_recup <= 0:
            logger.debug("Infeasible: Q_recup <= 0 -- pump outlet enthalpy exceeds T7 enthalpy")
            return None

        logger.debug(f"State 7: T={T7-273.15:.2f}C  h={h7:.1f} J/kg  Q_recup={Q_recup:.1f} J/kg")

        # ------------------------------------------------------------------
        # ECONOMISER  Q_econ fixed entirely from gas side
        # T_gas_out given -- cold-end pinch = T_gas_out - T7 (must be > 0)
        # ------------------------------------------------------------------
        T_gas_out     = cfg.T_gas_out
        T_gas_in      = cfg.T_gas_in
        pinch_cold    = T_gas_out - T7

        if pinch_cold <= 0:
            logger.debug(
                f"Infeasible: T_gas_out={T_gas_out-273.15:.2f}C <= T7={T7-273.15:.2f}C "
                f"-- cold-end pinch violated"
            )
            return None

        Q_econ_total  = cfg.mdot_gas * cfg.Cp_gas * (T_gas_in - T_gas_out)   # W
        q_econ_per_kg = Q_econ_total / cfg.mdot                                # J/kg

        if Q_econ_total <= 0:
            logger.debug("Infeasible: Q_econ <= 0 -- gas not providing heat")
            return None

        # ------------------------------------------------------------------
        # ITERATIVE SOLUTION FOR h1
        # h1 = h7 + q_econ_per_kg  (no circular dependency -- T7 is fixed by approach)
        # No iteration needed in this formulation.
        # But we keep the loop structure from the sample code for consistency,
        # to converge on h3 via the turbine-recuperator coupling:
        #   h2 depends on s1 which is fixed (h1 fixed)
        #   h3 = h2 - Q_recup  (Q_recup from cold side, fixed)
        #   Verify T3 >= T3_min and T3 superheated.
        # In fact with T7 approach-driven, h1 is explicit -- loop converges in 1 step.
        # We retain the structure for extensibility and to match sample code style.
        # ------------------------------------------------------------------
        h1_init = h7 + q_econ_per_kg
        h1      = h1_init

        converged        = False
        residual_history: List[float] = []

        for iteration in range(cfg.max_iterations):

            # State 1: turbine inlet (from economiser cold-out)
            T1 = td.temperature_from_PH(P_high, h1, fluid)
            s1 = td.entropy_from_PH(P_high, h1, fluid)

            # Hot-end pinch check
            pinch_hot = T_gas_in - T1
            if pinch_hot <= 0:
                logger.debug(
                    f"Iteration {iteration}: Infeasible: T_gas_in={T_gas_in-273.15:.2f}C "
                    f"<= T1={T1-273.15:.2f}C -- hot-end pinch violated"
                )
                return None

            # State 2: Turbine exhaust  -- h2s, h2, T2 vary with P_cond
            turb  = comp.turbine(h1, s1, P_cond, cfg.eta_turbine, fluid)
            h2    = turb["h"]
            T2    = turb["T"]
            h2s   = turb["h_isen"]

            if T2 <= T_sat:
                logger.debug(f"Iteration {iteration}: Infeasible: turbine exhaust two-phase")
                return None

            # State 3: Recuperator hot-out  h3 = h2 - Q_recup
            h3 = h2 - Q_recup
            T3 = td.temperature_from_PH(P_cond, h3, fluid)

            if T3 < cfg.T3_min:
                logger.debug(
                    f"Iteration {iteration}: Infeasible: T3={T3-273.15:.2f}C < "
                    f"T3_min={cfg.T3_min-273.15:.2f}C"
                )
                return None

            if T3 <= T_sat:
                logger.debug(f"Iteration {iteration}: Infeasible: T3 not superheated")
                return None

            if T2 <= T7:
                logger.debug(f"Iteration {iteration}: Infeasible: temperature cross in recuperator")
                return None

            # h1 update: h1 = h7 + q_econ -- already explicit, residual should be ~0
            h1_new   = h7 + q_econ_per_kg
            residual  = abs(h1_new - h1)
            residual_history.append(residual)

            logger.debug(
                f"Iteration {iteration:03d}: "
                f"T1={T1-273.15:.3f}C  T2={T2-273.15:.3f}C  "
                f"T3={T3-273.15:.3f}C  residual={residual:.4e} J/kg"
            )

            if residual < cfg.convergence_tol:
                converged = True
                logger.info(
                    f"Converged in {iteration+1} iteration(s) | "
                    f"residual={residual:.2e} J/kg"
                )
                break

            # Relaxed update
            h1 = cfg.relaxation_factor * h1_new + (1 - cfg.relaxation_factor) * h1

        if not converged:
            logger.warning(
                f"Failed to converge after {cfg.max_iterations} iterations. "
                f"Final residual: {residual_history[-1]:.2e} J/kg"
            )

        # ------------------------------------------------------------------
        # CONDENSER  state 3 -> state 5  using actual T3 (output)
        # ------------------------------------------------------------------
        cond_out = comp.condenser(T3, cfg.T5, P_cond, fluid)
        q_cond   = cond_out["Q_duty"]

        # ------------------------------------------------------------------
        # ENERGY BALANCE
        # ------------------------------------------------------------------
        w_turbine = turb["w"]
        w_pump    = pump_out["w"]
        w_net     = w_turbine - w_pump
        q_in      = q_econ_per_kg     # J/kg  external heat only

        if w_net <= 0 or q_in <= 0:
            logger.debug("Infeasible: non-positive net work or heat input")
            return None

        eta_cycle  = w_net / q_in
        mdot_cw    = (cfg.mdot * q_cond) / (4186.0 * cfg.dT_cw)

        logger.info(
            f"P_cond={P_cond/1e5:.4f} bar | "
            f"eta={eta_cycle*100:.3f}% | "
            f"W_net={cfg.mdot*w_net/1000:.2f} kW"
        )

        # ------------------------------------------------------------------
        # BUILD STATE POINTS
        # ------------------------------------------------------------------
        def make_state(label, T, P, h, desc=""):
            s = td.entropy_from_PH(P, h, fluid) if label != "5" else s5
            return StatePoint(label=label, T=T, P=P, h=h, s=s, description=desc)

        s2 = td.entropy_from_PH(P_cond, h2, fluid)
        s3 = td.entropy_from_PH(P_cond, h3, fluid)
        s6 = td.entropy_from_PH(P_high, h6, fluid)
        s7 = td.entropy_from_PH(P_high, h7, fluid)

        states = {
            1: StatePoint("1", T1, P_high, h1, s1, "Turbine inlet (economiser cold-out) [OUTPUT]"),
            2: StatePoint("2", T2, P_cond, h2, s2, "Turbine exhaust / Recup hot-in [varies with P_cond]"),
            3: StatePoint("3", T3, P_cond, h3, s3, "Recup hot-out / Condenser inlet [T3_min constraint]"),
            5: StatePoint("5", cfg.T5, P_cond, h5, s5, "Condenser outlet / Pump inlet [fixed]"),
            6: StatePoint("6", T6, P_high, h6, s6, "Pump outlet / Recup cold-in"),
            7: StatePoint("7", T7, P_high, h7, s7, "Recup cold-out / Econ cold-in [approach-driven]"),
        }

        # ------------------------------------------------------------------
        # COMPILE RESULTS
        # ------------------------------------------------------------------
        return CycleResults(
            states              = states,
            w_turbine_specific  = w_turbine,
            w_pump_specific     = w_pump,
            w_net_specific      = w_net,
            q_in_specific       = q_in,
            q_cond_specific     = q_cond,
            Q_recup_specific    = Q_recup,
            W_turbine           = cfg.mdot * w_turbine,
            W_pump              = cfg.mdot * w_pump,
            W_net               = cfg.mdot * w_net,
            Q_econ_total        = Q_econ_total,
            Q_cond_total        = cfg.mdot * q_cond,
            thermal_efficiency  = eta_cycle,
            mdot_cw             = mdot_cw,
            T_gas_in            = T_gas_in,
            T_gas_out           = T_gas_out,
            pinch_cold_end      = pinch_cold,
            pinch_hot_end       = pinch_hot,
            converged           = converged,
            iterations          = len(residual_history),
            residual_history    = residual_history,
        )

    # -------------------------------------------------------------------------
    def optimise(self) -> Tuple[Optional[CycleResults], Dict[float, CycleResults], float, float]:
        """
        Golden Section Search for P_cond that maximises cycle thermal efficiency.

        Feasible band from saturation physics:
          a = P_sat(T5)     lower bound  (T5 = condenser outlet must be subcooled)
          b = P_sat(T3_min) upper bound  (T3_min = condenser inlet must be superheated)

        Every evaluated pressure stored in visited{}.

        Returns
        -------
        best    : CycleResults at optimal P_cond
        visited : Dict {P_cond: CycleResults} for every feasible point evaluated
        a, b    : float  feasible band boundaries [Pa]
        """
        fluid  = self.fluid
        T3_min = self.config.T3_min
        T5     = self.config.T5

        a = td.sat_pressure(T5,     fluid)
        b = td.sat_pressure(T3_min, fluid)

        logger.info(
            f"Feasible band: "
            f"a=P_sat(T5={T5-273.15:.2f}C)={a/1e5:.5f} bar  "
            f"b=P_sat(T3_min={T3_min-273.15:.2f}C)={b/1e5:.5f} bar  "
            f"width={(b-a)/1e5:.5f} bar"
        )

        if a >= b:
            raise ValueError(
                f"No feasible band: "
                f"P_sat(T5={T5-273.15:.2f}C)={a/1e5:.4f} bar >= "
                f"P_sat(T3_min={T3_min-273.15:.2f}C)={b/1e5:.4f} bar. "
                f"Increase T3_min - T5 gap."
            )

        visited: Dict[float, CycleResults] = {}

        def objective(P: float) -> float:
            res = self.solve(P)
            if res is not None:
                visited[P] = res
                return -res.thermal_efficiency
            return float("inf")

        phi = (math.sqrt(5) - 1) / 2
        tol = 0.1   # Pa

        c = b - phi * (b - a)
        d = a + phi * (b - a)
        fc, fd = objective(c), objective(d)

        gss_iter = 0
        while (b - a) > tol:
            if fc < fd:
                b, d, fd = d, c, fc
                c  = b - phi * (b - a)
                fc = objective(c)
            else:
                a, c, fc = c, d, fd
                d  = a + phi * (b - a)
                fd = objective(d)
            gss_iter += 1

        P_optimal = (a + b) / 2.0
        best      = self.solve(P_optimal)

        if best is not None:
            visited[P_optimal] = best

        logger.info(
            f"GSS converged in {gss_iter} iterations | "
            f"P_optimal={P_optimal/1e5:.5f} bar | "
            f"eta={best.thermal_efficiency*100:.4f}% | "
            f"evaluations={len(visited)}"
        )

        return best, visited, a, b
