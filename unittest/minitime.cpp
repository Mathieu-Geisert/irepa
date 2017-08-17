#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>

int main(int argc, const char ** argv ){

  USING_NAMESPACE_ACADO;

  /// Pendulum hyperparameters
  const double l = 5;  // lever arm
  const double I = 0.5;   // joint inertia
  const double p0 = -.4;    // initial angle
  const double v0 = .0;    // initial velocity

  DifferentialState        p,v        ;     // the differential states
  Control                  u          ;     // the control input u
  Parameter                T;
  DifferentialEquation     f( 0.0, T );   // the differential equation

  //  -------------------------------------
  OCP ocp( 0.0, T, 20 );                        // time horizon of the OCP: [0,T]
  ocp.minimizeMayerTerm( T );

  f << dot(p) == v;                           // an implementation
  f << dot(v) == (u + l*sin(p)) / I;          // of the model equations

  ocp.subjectTo( f                       );   // minimize T s.t. the model,
  ocp.subjectTo( AT_START, p ==  p0 );     // the initial values for s,
  ocp.subjectTo( AT_START, v ==  v0 );     // v,

  ocp.subjectTo( AT_END  , p ==  0.0 );     // the terminal constraints for s
  //ocp.subjectTo( AT_END  , cos(p) ==  1.0 );     // the terminal constraints for s
  ocp.subjectTo( AT_END  , v ==  0.0 );     // and v,

  ocp.subjectTo( -8.0 <= v <=  8.0   );     // as well as the bounds on v
  ocp.subjectTo( -2. <= u <=  2.   );     // the control input u,
  ocp.subjectTo(  1. <= T <= 5.  );

  //  -------------------------------------

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm

  GnuplotWindow window;
  window.addSubplot( v, "THE VELOCITY v"      );
  window.addSubplot( u, "THE CONTROL INPUT u" );
  window.addSubplot( p, "THE ANGLE p"      );
  algorithm << window;

  algorithm.set( PRINTLEVEL, 2);
  algorithm.set( PRINT_COPYRIGHT, 0);
  algorithm.set( INTEGRATOR_TYPE,INT_RK78);
  algorithm.set( MAX_NUM_ITERATIONS, 50);

  algorithm.solve();                        // solves the problem.

  return 0;
}
