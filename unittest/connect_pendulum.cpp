/*
 * Connect in minimum time a starting position to a final state.
 */

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
    ("oparam",     po::value<std::string>()->default_value("/tmp/params.txt") , "Output parameters"  )
    ("ocontrol,c", po::value<std::string>()->default_value("/tmp/control.txt"), "Output controls")
    ("icontrol,i", po::value<std::string>()->default_value(""),                 "Input controls (guess)")
    ("istate,j",   po::value<std::string>()->default_value(""),                 "Input states (guess)")
    ("horizon,T",  po::value<double     >()->default_value(1.0),                "Horizon length")
    ("Tmin",       po::value<double     >()->default_value(2.0),                "Horizon length minimal")
    ("Tmax",       po::value<double     >()->default_value(0.1),                "Horizon length maximal")
    ("friction,K", po::value<double     >()->default_value(0.0),                "Friction coeff")
    ("decay,a",    po::value<double     >()->default_value(1.0),                "Cost decay rate")
    ("iter,N",     po::value<int        >()->default_value(100),                "Number of optim iterations")
    ("steps,d",    po::value<int        >()->default_value(20),                 "Discretization")
    ("printlevel", po::value<int        >()->default_value(0),                  "ACADO print level")
    ("initpos",    po::value<std::vector<double> >()->multitoken(),             "Initial position")
    ("initvel",    po::value<std::vector<double> >()->multitoken(),             "Initial velocity")
    ("finalpos",   po::value<std::vector<double> >()->multitoken(),           "Terminal position")
    ("finalvel",   po::value<std::vector<double> >()->multitoken(),           "Terminal velocity")
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
  const double & a_Tmin = vm["Tmin"].as<double>();
  const double & a_Tmax = vm["Tmax"].as<double>();
  const double & a_K = vm["friction"].as<double>();
  const double & a_decay = vm["decay"].as<double>();
  const int & a_d    = vm["steps"].as<int>();
  const int & a_N    = vm["iter"].as<int>();
  const int & a_shift= vm["shift"].as<int>();
  std::vector<double> a_p0,a_p1,a_v0,a_v1;
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

  a_p1 = vm["finalpos"].as< std::vector<double> >();
  a_p1.resize(1);
  a_v1 = vm["finalvel"].as< std::vector<double> >();
  a_v1.resize(1);


  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */

  /// Pendulum hyperparameters
  const double l = 4.905; // lever arm
  const double I = 0.25;   // joint inertia
  const double Kf = a_K;  // friction coeff
  const double UMAX = 2;

  DifferentialState        p,v    ;       // the differential states: pos, vel, control, 
  Control                  u      ;       // the control input u = a + b*t + .5*c*t*t
  Parameter                T;
  DifferentialEquation     f( 0.0, T );   // the differential equation

  //  -------------------------------------
  OCP ocp( 0.0, T, a_d );                        // time horizon of the OCP: [0,T]
  ocp.minimizeMayerTerm( T );

  f << dot(p)  == v;
  f << dot(v)  == (u - Kf*v + l*sin(p) ) /I;

  ocp.subjectTo( f                   );
  ocp.subjectTo( AT_START,  p ==  a_p0[0] );
  ocp.subjectTo( AT_START,  v ==  a_v0[0] );
  ocp.subjectTo( AT_END,    p ==  a_p1[0] );
  ocp.subjectTo( AT_END,    v ==  a_v1[0] );

  ocp.subjectTo(  a_Tmin <= T <= a_Tmax  );
  ocp.subjectTo( -UMAX  <= u <=  UMAX   );

  //  -------------------------------------

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm

  if( a_withPlot )
    {
      GnuplotWindow window;
      window.addSubplot( p, "Angle p"      );
      window.addSubplot( v, "Velocity v"      );
      window.addSubplot( u, "Control u" );
      algorithm << window;
    }

  algorithm.initializeControls(Us);
  algorithm.initializeDifferentialStates(Xs);

  {
    Grid timeGrid( 0.0, 1.0, a_d );
    VariablesGrid   Ps( 1, timeGrid );
    Ps(0,0) = a_T;
    algorithm.initializeParameters(Ps);
  }

  algorithm.set( PRINTLEVEL, vm["printlevel"].as<int>());
  algorithm.set( PRINT_COPYRIGHT, 0);
  algorithm.set( INTEGRATOR_TYPE,INT_RK45);
  algorithm.set( MAX_NUM_ITERATIONS, a_N);
  //algorithm.set( KKT_TOLERANCE, 1e-12);

  returnValue retval = algorithm.solve();                        // solves the problem.

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
  {
    VariablesGrid Ps;
    algorithm.getParameters( Ps );
    std::ofstream of(vm["oparam"  ].as<std::string>().c_str());
    Ps.print( of,"","","\n",10,10,"\t","\n");
  }

  
  std::cout << "RETURN CODE ["<<int(retval)<<"] "<< retval<<std::endl << "--" << std::endl; 
  return (int)retval;
}
