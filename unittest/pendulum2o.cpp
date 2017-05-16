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
  const double DT= a_T/a_d; 
  const double DT2= DT*DT;

  DifferentialState        p,v,u,du    ;     // the differential states: pos, vel, control, 
  Control                  a,b,c        ;    // the control input u = a + b*t + .5*c*t*t
  DifferentialEquation     f( 0.0, a_T );    // the differential equation

  //  -------------------------------------
  OCP ocp( 0.0, a_T, a_d );                        // time horizon of the OCP: [0,T]
  // ocp.minimizeLagrangeTerm( -cos(p)*5 + 1e-1*v*v + 1e-3*u*u );
  ocp.minimizeLagrangeTerm( p*p + 1e-3*v*v + 1e-3*u*u);
  // ocp.minimizeLagrangeTerm( p*p );

  f << dot(p) == v;                         // an implementation
  f << dot(v) == (u+a)/I + l*sin(p)/I;          // of the model equations
  f << dot(u) == b+du ;
  f << dot(du) == c ;


  ocp.subjectTo( f                   );         // minimize intergral cost
  ocp.subjectTo( AT_START,  p ==  a_p0[0] );     // Initial value for p
  ocp.subjectTo( AT_START,  v ==  a_v0[0] );     // Initial value for v
  ocp.subjectTo( AT_START,  u ==  0 );           // a(0) and u(0) are redundant: impose u(0)=0 
  ocp.subjectTo( AT_START, du ==  0 );           // b(0) and du(0) are redundant: impose du(0)=0 

  // ocp.subjectTo( AT_END  , cos(p) ==  1.0 );    // the terminal constraints for p mod pi
  // ocp.subjectTo( AT_END  , p ==  0.0 );         // the terminal constraints for p
  // ocp.subjectTo( AT_END  , v ==  0.0 );         // the terminal constraints for v
  // ocp.subjectTo( AT_END  , a+u ==  0 );        // Steady terminal state

  ocp.subjectTo( -8.0 <= v <=  8.0   );     // Bounds on v

  // Bounds on u
  ocp.subjectTo( -2.  <= a+u <=  2.   );                                  // U bound at init grid point
  ocp.subjectTo( -2.  <= a+u+(b+du)*DT+.5*c*DT2 <=  2.   );               // U bound at final grid point
  ocp.subjectTo( -2.  <= a+u+(b+du)*DT*.5 +.5*c*DT2*.5 *.5  <=  2.   );   // U bound at middle grid point

  ocp.subjectTo( -2.  <= a+u+(b+du)*DT*.25+.5*c*DT2*.25*.25 <=  2.   );   // U bound at 25% grid point
  ocp.subjectTo( -2.  <= a+u+(b+du)*DT*.75+.5*c*DT2*.75*.75 <=  2.   );   // U bound at 75% grid point

  //  -------------------------------------

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm

  if( a_withPlot )
    {
      GnuplotWindow window;
      window.addSubplot( p, "Angle p"      );
      window.addSubplot( v, "Velocity v"      );
      window.addSubplot( a+u, "Control u" );
      algorithm << window;
    }

  algorithm.initializeControls(Us);
  algorithm.initializeDifferentialStates(Xs);

  algorithm.set( PRINTLEVEL, vm["printlevel"].as<int>());
  algorithm.set( PRINT_COPYRIGHT, 0);
  algorithm.set( INTEGRATOR_TYPE,INT_RK78);
  algorithm.set( MAX_NUM_ITERATIONS, a_N);
  //algorithm.set( KKT_TOLERANCE, 1e-12);

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

  return 0;
}
