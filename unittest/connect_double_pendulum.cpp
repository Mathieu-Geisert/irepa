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
    ("friction",   po::value<std::vector<double> >()->multitoken()
     ->default_value(std::vector<double>{0.,0.}, "0. 0."),                      "Friction coeff")
    //("friction,K", po::value<double     >()->default_value(0.0),                "Friction coeff")
    ("decay,a",    po::value<double     >()->default_value(1.0),                "Cost decay rate")
    ("iter,N",     po::value<int        >()->default_value(100),                "Number of optim iterations")
    ("steps,d",    po::value<int        >()->default_value(20),                 "Discretization")
    ("printlevel", po::value<int        >()->default_value(0),                  "ACADO print level")
    ("initpos",    po::value<std::vector<double> >()->multitoken(),             "Initial position")
    ("initvel",    po::value<std::vector<double> >()->multitoken(),             "Initial velocity")
    ("finalpos",   po::value<std::vector<double> >()->multitoken(),             "Terminal position")
    ("finalvel",   po::value<std::vector<double> >()->multitoken(),             "Terminal velocity")
    ("umax",       po::value<std::vector<double> >()->multitoken()
     ->default_value(std::vector<double>{10.,10.}, "10 10"),                     "Torque limit")
    ("shift,t",    po::value<int>()->default_value(0),                           "Number of time shifts")
    ("armature",    po::value<double>()->default_value(0.),                         "Joint armature")
    ("statefromfile,f", "Init state from file")
    ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc,
                                   po::command_line_style::unix_style 
                                   ^ po::command_line_style::allow_short), vm);
  po::notify(vm);    
  
  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  const double & a_T = vm["horizon"].as<double>();
  const double & a_Tmin = vm["Tmin"].as<double>();
  const double & a_Tmax = vm["Tmax"].as<double>();
  //  const double & a_K = vm["friction"].as<double>();
  const double & a_decay = vm["decay"].as<double>();
  const int & a_d    = vm["steps"].as<int>();
  const int & a_N    = vm["iter"].as<int>();
  const int & a_shift= vm["shift"].as<int>();
  std::vector<double> a_p0,a_p1,a_v0,a_v1,a_umax,a_K;
  const bool a_withPlot = vm.count("plot")!=0;
  const std::string & a_guessCFile = vm["icontrol"].as<std::string>();
  const std::string & a_guessSFile = vm["istate"].as<std::string>();
  const double & a_armature = vm["armature"].as<double>();

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
      a_p0.resize(2);
      a_v0 = vm["initvel"].as< std::vector<double> >();
      a_v0.resize(2);
    }
  else // Init config from state file.
    {
      std::cout << "Auto init pos = " << Xs(0,0) << ", " << Xs(0,1) << std::endl;
      a_p0.resize(2); a_p0[0] = Xs(0,0);
      a_v0.resize(2); a_v0[0] = Xs(0,1);
    }

  a_p1 = vm["finalpos"].as< std::vector<double> >();
  a_p1.resize(2);
  a_v1 = vm["finalvel"].as< std::vector<double> >();
  a_v1.resize(2);
  a_umax = vm["umax"].as< std::vector<double> >();
  a_umax.resize(2);
  a_K = vm["friction"].as< std::vector<double> >();
  a_K.resize(2);


  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */

  /// Pendulum hyperparameters
  const double p  =   .5;    // lever arm
  const double m  =  3. ;    // mass
  const double a  =  2. ;    // armature
  const double g  = 9.81;    // gravity constant
  const double Kf0 = a_K[0], Kf1 = a_K[1];     // friction coeff
  const double DT = a_T/a_d; // integration time
  const double U0MAX = a_umax[0], U1MAX = a_umax[1];

  DifferentialState        q0,q1,vq0,vq1;
  Control                  u0,u1  ;       // the control input u = a + b*t + .5*c*t*t
  Parameter                T;
  DifferentialEquation     f( 0.0, T );   // the differential equation

  //  -------------------------------------
  OCP ocp( 0.0, T, a_d );                        // time horizon of the OCP: [0,T]
  ocp.minimizeMayerTerm( T );

  IntermediateState nM = (16*a*a + 16*a*m*p*p*cos(q1) + 28*a*m*p*p 
                          + 4*m*m*p*p*p*p*sin(q1)*sin(q1) + m*m*p*p*p*p);
  IntermediateState 
    Mi00 = (  16*a + 4*m*p*p)/nM,
    Mi01 = ( -4*m*p*p*(2*cos(q1) + 1))/nM,
    Mi11 = (16*a + 8*m*p*p*(2*cos(q1) + 3))/nM,
    b0    = -m*p*(3*g*sin(q0) + g*sin(q0 + q1) + 2*p*vq0*vq1*sin(q1) + p*vq1*vq1*sin(q1))/2,
    b1    =    m*p*(-g*sin(q0 + q1) + p*vq0*vq0*sin(q1))/2;
 
  IntermediateState 
    tau0 = u0 - Kf0*vq0 - b0,
    tau1 = u1 - Kf1*vq1 - b1;

  f << dot(q0)  == vq0;
  f << dot(vq0) == Mi00*tau0 + Mi01*tau1;
  f << dot(q1)  == vq1;
  f << dot(vq1) == Mi01*tau0 + Mi11*tau1;

  ocp.subjectTo( f                   );

  ocp.subjectTo( AT_START,  q0  ==  a_p0[0] );
  ocp.subjectTo( AT_START,  vq0 ==  a_v0[0] );
  ocp.subjectTo( AT_START,  q1  ==  a_p0[1] );
  ocp.subjectTo( AT_START,  vq1 ==  a_v0[1] );

  ocp.subjectTo( AT_END,    q0  ==  a_p1[0] );
  ocp.subjectTo( AT_END,    vq0 ==  a_v1[0] );
  ocp.subjectTo( AT_END,    q1  ==  a_p1[1] );
  ocp.subjectTo( AT_END,    vq1 ==  a_v1[1] );

  ocp.subjectTo(  a_Tmin <= T <= a_Tmax  );
  ocp.subjectTo( -U0MAX  <= u0 <=  U0MAX   );
  ocp.subjectTo( -U1MAX  <= u1 <=  U1MAX   );

  ocp.subjectTo( -M_PI  <= q1 <=  M_PI   );

  //  -------------------------------------

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm

  if( a_withPlot )
    {
      GnuplotWindow window;
      window.addSubplot( q0, "Angle q0"      );
      window.addSubplot( q1, "Angle q1"      );
      window.addSubplot( u0, "Control u0" );
      window.addSubplot( u1, "Control u1" );
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
