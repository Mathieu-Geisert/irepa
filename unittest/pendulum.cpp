#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>

#include <boost/program_options.hpp>
namespace po = boost::program_options;


void f()
{
}


int main(int argc, const char ** argv ){

  USING_NAMESPACE_ACADO;


  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("ostate,s",   po::value<std::string>()->default_value("/tmp/states.txt") , "Output states"  )
    ("ocontrol,c", po::value<std::string>()->default_value("/tmp/control.txt"), "Output controls")
    ("icontrol,i", po::value<std::string>()->default_value("/tmp/control.txt"), "Input controls (guess)")
    ("horizon,T",  po::value<double     >()->default_value(5.0),                "Horizon length")
    ("steps,d",    po::value<double     >()->default_value(20),                 "Discretization")
    ("init-pose,p",po::value<std::vector<double> >(),                           "Initial position")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    
  
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  if (vm.count("compression")) {
    std::cout << "Compression level was set to " 
         << vm["compression"].as<int>() << ".\n";
  } else {
    std::cout << "Compression level was not set.\n";
  }


  /// Pendulum hyperparameters
  const double l = 4.905; // lever arm
  const double I = 0.45;   // joint inertia
  const double T = 5.;

  DifferentialState        p,v        ;     // the differential states
  Control                  u          ;     // the control input u
  Parameter                p0         ;     // the time horizon T
  DifferentialEquation     f( 0.0, T );     // the differential equation


  //  -------------------------------------
  OCP ocp( 0.0, T, 100 );                        // time horizon of the OCP: [0,T]
  ocp.minimizeLagrangeTerm( -cos(p)*10 + 0.1*v*v + 1e-3*u*u );

  f << dot(p) == v;                         // an implementation
  f << dot(v) == u/I + l*sin(p)/I;          // of the model equations

  ocp.subjectTo( f                   );     // minimize T s.t. the model,
  ocp.subjectTo( AT_START, p ==  3. );     // the initial values for s,
  ocp.subjectTo( AT_START, v ==  0.0 );     // v,

  //ocp.subjectTo( AT_END  , cos(p) ==  1.0 );     // the terminal constraints for s
  //ocp.subjectTo( AT_END  , v ==  0.0 );     // and v,

  ocp.subjectTo( -8.0 <= v <=  8.0   );     // as well as the bounds on v
  ocp.subjectTo( -2. <= u <=  2.   );     // the control input u,
  //  -------------------------------------

  GnuplotWindow window;
  window.addSubplot( v, "THE VELOCITY v"      );
  window.addSubplot( u, "THE CONTROL INPUT u" );
  window.addSubplot( p, "THE ANGLE p"      );

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm
  algorithm << window;
  //algorithm.set( KKT_TOLERANCE, 1e-12 );
  //algorithm.set( INTEGRATOR_TYPE, INT_RK78);
  //algorithm.set( MAX_NUM_ITERATIONS, 2000);
  algorithm.solve();                        // solves the problem.

  algorithm.getDifferentialStates("pendulum-s.txt");
  algorithm.getControls("pendulum-c.txt");

  return 0;
}
