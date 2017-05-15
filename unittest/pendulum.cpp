#include <acado_toolkit.hpp>
#include <acado_gnuplot.hpp>

#include <boost/program_options.hpp>
namespace po = boost::program_options;



int main(int argc, const char ** argv ){

  USING_NAMESPACE_ACADO;

  /* --- OPTIONS ----------------------------------------------------------------------------- */
  /* --- OPTIONS ----------------------------------------------------------------------------- */
  /* --- OPTIONS ----------------------------------------------------------------------------- */
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h",     "produce help message")
    ("plot",       "with plots")
    ("ostate,s",   po::value<std::string>()->default_value("/tmp/states.txt") , "Output states"  )
    ("ocontrol,c", po::value<std::string>()->default_value("/tmp/control.txt"), "Output controls")
    ("icontrol,i", po::value<std::string>()->default_value(""),                 "Input controls (guess)")
    ("istate,j",   po::value<std::string>()->default_value(""),                 "Input states (guess)")
    ("horizon,T",  po::value<double     >()->default_value(5.0),                "Horizon length")
    ("iter,N",     po::value<int        >()->default_value(100),                "Number of optim iterations")
    ("steps,d",    po::value<int        >()->default_value(20),                 "Discretization")
    ("printlevel", po::value<int        >()->default_value(0),                  "ACADO print level")
    ("initpos,p",po::value<std::vector<double> >()->multitoken(),               "Initial position")
    ("initvel,v",po::value<std::vector<double> >()->multitoken(),               "Initial velocity")
    ("shift,t",    po::value<int>()->default_value(0),                          "Number of time shifts")
    ("statefromfile,f", "Init state from file")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    
  
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  const double & a_T = vm["horizon"].as<double>();
  const int & a_d    = vm["steps"].as<int>();
  const int & a_N    = vm["iter"].as<int>();
  const int & a_shift= vm["shift"].as<int>();
  std::vector<double> a_p0;
  std::vector<double> a_v0;
  const bool a_withPlot = vm.count("plot")!=0;
  const std::string & a_guessCFile = vm["icontrol"].as<std::string>();
  const std::string & a_guessSFile = vm["istate"].as<std::string>();

  VariablesGrid Us,Xs;

  if(a_guessCFile.size()>0)
    {
      Us.read(a_guessCFile.c_str());
      for(int loop=0;loop<a_shift;++loop) Us.shiftBackwards();
    }
  if(a_guessSFile.size()>0)
    {
      Xs.read(a_guessSFile.c_str());
      for(int loop=0;loop<a_shift;++loop) Xs.shiftBackwards();
    }

  if (vm.count("statefromfile")==0) // Init config explicit from option.
    {
      a_p0 = vm["initpos"].as< std::vector<double> >();
      a_p0.resize(1);
      a_v0 = vm["initvel"].as< std::vector<double> >();
      a_v0.resize(1);
    }
  else // Init config from state file.
    {
      std::cout << "Auto init pos = " << Xs(0,0) << ", " << Xs(0,1) << std::endl;
      a_p0.resize(1); a_p0[0] = Xs(0,0);
      a_v0.resize(1); a_v0[0] = Xs(0,1);
    }

  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */

  /// Pendulum hyperparameters
  const double l = 4.905; // lever arm
  const double I = 0.45;   // joint inertia

  DifferentialState        p,v        ;     // the differential states
  Control                  u          ;     // the control input u
  DifferentialEquation     f( 0.0, a_T );   // the differential equation

  //  -------------------------------------
  OCP ocp( 0.0, a_T, a_d );                        // time horizon of the OCP: [0,T]
  //ocp.minimizeLagrangeTerm( -cos(p)*5 + 1e-1*v*v + 1e-3*u*u );
  ocp.minimizeLagrangeTerm( p*p + 1e-3*v*v + 1e-3*u*u);
  //ocp.minimizeLagrangeTerm( p*p );

  f << dot(p) == v;                         // an implementation
  f << dot(v) == u/I + l*sin(p)/I;          // of the model equations

  ocp.subjectTo( f                   );     // minimize T s.t. the model,
  ocp.subjectTo( AT_START, p ==  a_p0[0] );     // the initial values for s,
  ocp.subjectTo( AT_START, v ==  a_v0[0] );     // v,

  //ocp.subjectTo( AT_END  , cos(p) ==  1.0 );     // the terminal constraints for s
  //ocp.subjectTo( AT_END  , p ==  0.0 );     // the terminal constraints for s
  //ocp.subjectTo( AT_END  , v ==  0.0 );     // and v,

  ocp.subjectTo( -8.0 <= v <=  8.0   );     // as well as the bounds on v
  ocp.subjectTo( -2. <= u <=  2.   );     // the control input u,
  //  -------------------------------------


  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm

  if( a_withPlot )
    {
      GnuplotWindow window;
      window.addSubplot( v, "THE VELOCITY v"      );
      window.addSubplot( u, "THE CONTROL INPUT u" );
      window.addSubplot( p, "THE ANGLE p"      );
      algorithm << window;
    }

  //if(a_guessCFile.size()>0) algorithm.initializeControls           (a_guessCFile.c_str());
  //if(a_guessSFile.size()>0) algorithm.initializeDifferentialStates (a_guessSFile.c_str());
  algorithm.initializeControls(Us);
  algorithm.initializeDifferentialStates(Xs);

  algorithm.set( PRINTLEVEL, vm["printlevel"].as<int>());
  algorithm.set( PRINT_COPYRIGHT, 0);
  algorithm.set( INTEGRATOR_TYPE,INT_RK78);
  algorithm.set( MAX_NUM_ITERATIONS, a_N);



  
  // //const double Kp=4.0,Kv=4.0;
  // //for(int i=0;i<20;++i)
  // //Us(i,0) = -Kp*Xs(i,0) - Kv*Xs(i,1);
  // IntegratorRK78 integrator(f);
  // double x0[2] = {0.,1.};
  // double u0[1] = {0.};
  // integrator.integrate(0.,0.1,x0,0,0,u0);
  // VariablesGrid differentialStates;
  // integrator.getX( differentialStates );
  // std::cout << "*** x = " << differentialStates[0] << std::endl;
  // //integrator.integrate( t_start, t_end, x_start, 0, p, u );




  algorithm.solve();                        // solves the problem.

  {
    VariablesGrid Us;
    algorithm.getControls( Us );
    std::ofstream of(vm["ocontrol"  ].as<std::string>().c_str());
    Us.print( of,"","","\n",10,10,"\t","\n");
  }
  {
    VariablesGrid Xs;
    algorithm.getDifferentialStates( Xs );
    std::ofstream of(vm["ostate"  ].as<std::string>().c_str());
    Xs.print( of,"","","\n",10,10,"\t","\n");
  }
  // algorithm.getAlgebraicStates("/tmp/astates.txt");
  // algorithm.getParameters("/tmp/param.txt");

  return 0;
}
