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
    ("plot",       po::value<int>()->default_value(-1),                         "with plots")
    ("ostate,s",   po::value<std::string>()->default_value("/tmp/states.txt") , "Output states"  )
    ("ocontrol,c", po::value<std::string>()->default_value("/tmp/control.txt"), "Output controls")
    ("icontrol,i", po::value<std::string>()->default_value(""),                 "Input controls (guess)")
    ("istate,j",   po::value<std::string>()->default_value(""),                 "Input states (guess)")
    ("horizon,T",  po::value<double     >()->default_value(5.0),                "Horizon length")
    ("friction,K", po::value<double     >()->default_value(0.0),                "Friction coeff")
    ("decay,a",    po::value<double     >()->default_value(1.0),                "Cost decay rate")
    ("iter,N",     po::value<int        >()->default_value(100),                "Number of optim iterations")
    ("steps,d",    po::value<int        >()->default_value(20),                 "Discretization")
    ("printlevel", po::value<int        >()->default_value(0),                  "ACADO print level")
    ("initpos,p",po::value<std::vector<double> >()->multitoken(),               "Initial position")
    ("initvel,v",po::value<std::vector<double> >()->multitoken(),               "Initial velocity")
    ("shift,t",    po::value<int>()->default_value(0),                          "Number of time shifts")
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
  const double & a_K = vm["friction"].as<double>();
  const double & a_decay = vm["decay"].as<double>();
  const int & a_d    = vm["steps"].as<int>();
  const int & a_N    = vm["iter"].as<int>();
  const int & a_shift= vm["shift"].as<int>();
  std::vector<double> a_q0;
  std::vector<double> a_v0;
  const int a_plotref = vm["plot"].as<int>();
  const bool a_withPlot = a_plotref>=0;
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
      a_q0 = vm["initpos"].as< std::vector<double> >();
      a_q0.resize(2);
      std::cout << "Init pos = " << a_q0[0] << " " << a_q0[1] << std::endl;
      a_v0 = vm["initvel"].as< std::vector<double> >();
      a_v0.resize(2);
    }
  else // Init config from state file.
    {
      std::cout << "Auto init pos = " << Xs(0,0) << ", " << Xs(0,1) << std::endl;
      a_q0.resize(2); a_q0[0] = Xs(0,0); a_q0[1] = Xs(0,1);
      a_v0.resize(2); a_v0[0] = Xs(0,2); a_v0[1] = Xs(0,3);
    }

  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */

  /// Pendulum hyperparameters
  const double p  =   .5;    // lever arm
  const double m  =  3. ;    // mass
  const double a  = 10. ;    // armature
  const double g  = 9.81;    // gravity constant
  const double Kf = a_K;     // friction coeff
  const double DT = a_T/a_d; // integration time
  const double umax = 15.;   // torque limit

  DifferentialState        q0,q1,vq0,vq1   ;     // the differential states
  Control                  u0,u1         ;     // the control input u
  DiscretizedDifferentialEquation     f( DT );   // the differential equation
  TIME                     t;

  //  -------------------------------------
  OCP ocp( 0.0, a_T, a_d );                        // time horizon of the OCP: [0,T]
  IntermediateState cost = (q0*q0 + q1*q1);
  ocp.minimizeLagrangeTerm(cost);

  // denom = 16*a**2 + 16*a*m*p**2*cos(q1) + 28*a*m*p**2 + 4*m**2*p**4*sin(q1)**2 + m**2*p**4
  // Mi    =  [          16*a + 4*m*p**2,       -4*m*p**2*(2*cos(q1) + 1)],
  //          [-4*m*p**2*(2*cos(q1) + 1), 16*a + 8*m*p**2*(2*cos(q1) + 3)]] / denom
  // b    = [-m*p*(3*g*sin(q0) + g*sin(q0 + q1) + 2*p*vq0*vq1*sin(q1) + p*vq1**2*sin(q1))/2],
  //        [                                    m*p*(-g*sin(q0 + q1) + p*vq0**2*sin(q1))/2]]
  // qdd = Minv(u-Kv-b)

  IntermediateState nM = (16*a*a + 16*a*m*p*p*cos(q1) + 28*a*m*p*p 
                          + 4*m*m*p*p*p*p*sin(q1)*sin(q1) + m*m*p*p*p*p);
  IntermediateState 
    Mi00 = (  16*a + 4*m*p*p)/nM,
    Mi01 = ( -4*m*p*p*(2*cos(q1) + 1))/nM,
    Mi11 = (16*a + 8*m*p*p*(2*cos(q1) + 3))/nM,
    //b0   = -m*p*(2*g*sin(q0) + g*sin(q0 + q1) + 2*p*vq0*vq1*sin(q1) + p*vq1*vq1*sin(q1)),
    //b1   =  m*p*(-g*sin(q0 + q1) + p*vq0*vq0*sin(q1));
    b0    = -m*p*(3*g*sin(q0) + g*sin(q0 + q1) + 2*p*vq0*vq1*sin(q1) + p*vq1*vq1*sin(q1))/2,
    b1    =    m*p*(-g*sin(q0 + q1) + p*vq0*vq0*sin(q1))/2;
 

  IntermediateState 
    tau0 = u0 - Kf*vq0 - b0,
    tau1 = u1 - Kf*vq1 - b1;

  IntermediateState
    a0   = Mi00*tau0 + Mi01*tau1,
    a1   = Mi01*tau0 + Mi11*tau1;

  f << next(q0)  == q0 + vq0*DT + a0*DT*DT;
  f << next(vq0) == vq0 + a0*DT;
  f << next(q1)  == q1 + vq1*DT + a1*DT*DT;
  f << next(vq1) == vq1 + a1*DT;

  ocp.subjectTo( f                   );     
  ocp.subjectTo( AT_START, q0  ==  a_q0[0] );
  ocp.subjectTo( AT_START, vq0 ==  a_v0[0] );
  ocp.subjectTo( AT_START, q1  ==  a_q0[1] );
  ocp.subjectTo( AT_START, vq1 ==  a_v0[1] );

  ocp.subjectTo( -umax <= u0 <=  umax   );     // the control input u,
  ocp.subjectTo( -umax <= u1 <=  umax   );     // the control input u,

  //  -------------------------------------

  OptimizationAlgorithm algorithm(ocp);     // the optimization algorithm

  GnuplotWindow window0;
  if( a_withPlot )
    {
      if( a_plotref == 0)
        {
          window0.addSubplot( q0, "THE ANGLE q0"      );
          window0.addSubplot( vq0, "THE VELOCITY v0"      );
          window0.addSubplot( u0, "THE CONTROL INPUT u0" );
        }
      else if( a_plotref == 1)
        {
          window0.addSubplot( q1, "THE ANGLE q1"      );
          window0.addSubplot( vq1, "THE VELOCITY v1"      );
          window0.addSubplot( u1, "THE CONTROL INPUT u1" );
        }

      window0.addSubplot( cost, "COST" );
      algorithm << window0;
    }

  algorithm.initializeControls(Us);
  algorithm.initializeDifferentialStates(Xs);

  algorithm.set( PRINTLEVEL, vm["printlevel"].as<int>());
  algorithm.set( PRINT_COPYRIGHT, 0);
  //algorithm.set( INTEGRATOR_TYPE,INT_RK78);
  algorithm.set( MAX_NUM_ITERATIONS, a_N);

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
