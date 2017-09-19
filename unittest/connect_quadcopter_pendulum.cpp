
/*
 * Connect in minimum time a starting position to a final state.
 */

#include <acado_toolkit.hpp>
#include "pycado/utils.hpp"

struct OptionsQuadcopter : public OptionsOCP
{
  virtual void addExtraOptions()
  {
    desc.add_options()
      ("maxAngle",       po::value<double>()->default_value(1.7),         "Max angles for roll and pitch");
    desc.add_options()
      ("maxAngleSing",   po::value<double>()->default_value(1.4),         "Max angle for the singular axis i.e pitch");
    desc.add_options()
      ("sphericalObstacle",   po::value<std::vector<double> >()->multitoken(),     "Spherical obstacle: size, position x, position y, position z.");
    desc.add_options()
      ("maxAnglePend",   po::value<double>()->default_value(1.0),         "Max angle for the pendulum");
  }
  const std::vector<double> sphericalObstacle()      { return vm["sphericalObstacle"].as< std::vector<double> >(); }
  const double & maxAngle()        { return vm["maxAngle"].as<double>(); }
  const double & maxAngleSing()        { return vm["maxAngleSing"].as<double>(); }
  const double & maxAnglePend()        { return vm["maxAnglePend"].as<double>(); }
  bool withSphericalObstacle()   { return vm.count("sphericalObstacle")>0; }
};

void outputControlAndStateWithoutLagrangeTerm( ACADO::OptimizationAlgorithm  & algorithm, OptionsOCP & opts )
{
  USING_NAMESPACE_ACADO;
  if( opts.withOutputControl() )
    {
      VariablesGrid Us;
      algorithm.getControls( Us );
      std::ofstream of(opts.outputControlFile().c_str());
      Us.print( of,"","","\n",10,10,"\t","\n");
    }

  if( opts.withOutputState() )
    {
      VariablesGrid Xs_full;
      algorithm.getDifferentialStates( Xs_full );
      VariablesGrid Xs(10, Xs_full.getTimePoints());
      int idex = 0;
      for (int i=0; i<17; i++) //Do not copy last variable = LagrangeTerm
      {
          if (i==5 || i==14 || i==8 ) continue; //Do not copy yaw, r and pendYaw
          for (int j=0; j<Xs_full.getTimePoints().getNumPoints(); j++)
          {
              if (i==15)
              {
                  Xs(j,idex) = cos(Xs_full(j, 8))/cos(Xs_full(j, 7))*Xs_full(j, 15)-sin(Xs_full(j, 8))/cos(Xs_full(j, 7))*Xs_full(j, 16);
              }
              if (i==16)
              {
                  Xs(j,idex) = sin(Xs_full(j, 8))*Xs_full(j, 15)+cos(Xs_full(j, 8))*Xs_full(j, 16);
              }
              else
              {
                  Xs(j, idex) = Xs_full(j, i);
              }
          }
          idex++;
      }
      std::ofstream of(opts.outputStateFile().c_str());
      Xs.print( of,"","","\n",10,10,"\t","\n");
    }
}

void initControlAndStateQuadcopter( ACADO::OptimizationAlgorithm  & algorithm, OptionsOCP & opts )
{
    USING_NAMESPACE_ACADO;
    if(opts.withGuessControl()) {  algorithm.initializeControls          (opts.guessControl());}
    if(opts.withGuessState  ())
    {
        VariablesGrid Xs(opts.guessState());
        VariablesGrid Xs_full(18, Xs.getTimePoints());
        Xs_full.setAll(0);
        for (int i=0; i<Xs_full.getTimePoints().getNumPoints(); i++)
        {
            Xs_full(i,0) = Xs(i,0);
            Xs_full(i,1) = Xs(i,1);
            Xs_full(i,2) = Xs(i,2);
            Xs_full(i,3) = Xs(i,3);
            Xs_full(i,4) = Xs(i,4);
            Xs_full(i,6) = Xs(i,5);
            Xs_full(i,7) = Xs(i,6);
            Xs_full(i,9) = Xs(i,7);
            Xs_full(i,10) = Xs(i,8);
            Xs_full(i,11) = Xs(i,9);
            Xs_full(i,12) = Xs(i,10);

        }
        algorithm.initializeDifferentialStates(Xs_full);
    }
}



int main(int argc, const char ** argv )
{
  USING_NAMESPACE_ACADO;
  Timer timer;



  /* --- OPTIONS ----------------------------------------------------------------------------- */
  /* --- OPTIONS ----------------------------------------------------------------------------- */
  /* --- OPTIONS ----------------------------------------------------------------------------- */
  OptionsQuadcopter opts; opts.parse(argc,argv);
  opts.NQ = 5; opts.NV = 5; opts.NU = 4;
  assert( opts.friction().size() == 2);
  assert( opts.umax()    .size() == 4);

  opts.displayBoundaryConditions(std::cout);

  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */
  /* --- OCP ----------------------------------------------------------------------------- */

  /// Quadcopter hyperparameters
  const double l     =  .5 ;    // lever arm
  const double mq     = 2.5 ;   // mass of quadrotor
  const double Ixx   = 1.  ;    // roll inertia XX
  const double Iyy   = 1.  ;    // pitch inertia YY
  const double Izz   = 1.6 ;    // yaw inertia ZZ
  const double Ct    = 0.1 ;    // Coeff torque/force of each propeller
  const double g     = 9.81;    // gravity constant
  const double DT    = opts.T()/opts.steps(); // integration time
  const double umax1 = opts.umax()[0], umax2 = opts.umax()[1], umax3 = opts.umax()[2], umax4 = opts.umax()[3];

  /// Pendulum hyperparameters
  const double Lpend = 2. ;     // length of the pendulum
  const double mp    = 0.3;     // Mass of the pendulum

  ///Whole system hyperparameters
  // We consider here that the
  const double m = mp + mq;     // Total mass
  const double Xqg = mp/m*Lpend; // Distance quadcopter to CoM
  const double Xpg = mq/m*Lpend; // Distance pendulum to CoM
  const double Ipend = pow(Xqg, 2)*mq + pow(Xpg, 2)*mp; //Inertia aournd x and y of the whole system at the CoM



  //WARNING: The angles roll-pitch-yaw are considered in the inverse order of the usual convention RPY
  //         i.e YAW then PITCH then ROLL. This means that ROLL always affect the world's Y axis, no matter the yaw.
  //
  // p, q, r follow the usual convention i.e p is the rotational speed around the local X axis,
  //                                         q is the rotational speed around the local Y axis,
  //                                         r is the rotational speed around the local Z axis,

  // Here qx. qy. qz, vx, vy, vz represent the position/velocity of the CoM
  // roll, pitch, yaw reprensent the orientation of only the quadcopter with respect to the WORLD.
  // pendRoll, pendPitch are also used using YAW then PITCH the ROLL convention.
  // pendRoll, pendPitch represent the orientation of the pendulum with respect to the WORLD.
  // So the angle between the pendulum and the quadrotor are not represented in this model...

  DifferentialState        qx, qy, qz, roll, pitch, yaw, pendRoll, pendPitch, pendYaw, vx, vy, vz, p, q, r, Wpendx, Wpendy;
  Control                  f1,f2,f3,f4;
  Parameter                T;
  DifferentialEquation     f( 0.0, T );   // the differential equation

  IntermediateState dyaw = cos(yaw)*tan(pitch)*p+sin(yaw)*tan(pitch)*q+r;
  IntermediateState dpitch = sin(yaw)*p+cos(yaw)*q;
  IntermediateState droll = cos(yaw)/cos(pitch)*p-sin(yaw)/cos(pitch)*q;
  IntermediateState fTot = f1+f2+f3+f4;
  IntermediateState dPendYaw = cos(pendYaw)*tan(pendPitch)*Wpendx+sin(pendYaw)*tan(pendPitch)*Wpendy;
  IntermediateState dPendPitch = sin(pendYaw)*Wpendx+cos(pendYaw)*Wpendy;
  IntermediateState dPendRoll = cos(pendYaw)/cos(pendPitch)*Wpendx-sin(pendYaw)/cos(pendPitch)*Wpendy;

  //  --- SETUP OCP -----------------------
  OCP ocp( 0.0, T, opts.steps() );                        // time horizon of the OCP: [0,T]

  ocp.minimizeMayerTerm( T );
  ocp.minimizeLagrangeTerm( 1000*square(yaw) + 100*square(dyaw) );      //WARNING: LagrangeTerm is added as a additionnal state...

  // DIFFERENTIAL EQUATIONS
  // Quadrotor
  f << dot(qx) == vx;
  f << dot(qy) == vy;
  f << dot(qz) == vz;
  f << dot(vx) == fTot*sin(pitch)/m;                //(f1+f2+f3+f4)*sin(pitch)/m;
  f << dot(vy) == fTot*sin(roll)*cos(pitch)/m;      //(f1+f2+f3+f4)*sin(roll)*cos(pitch)/m;
  f << dot(vz) == fTot*cos(roll)*cos(pitch)/m - g;  //(f1+f2+f3+f4)*cos(roll)*cos(pitch)/m - g;
  f << dot(yaw) == dyaw;                            //-cos(yaw)*tan(pitch)*p+sin(yaw)*tan(pitch)*q+r;
  f << dot(pitch) == dpitch;                        //sin(yaw)*p+cos(yaw)*q;
  f << dot(roll) == droll;                          //cos(yaw)/cos(pitch)*p-sin(yaw)/cos(pitch)*q;
  f << dot(p) == (l*(f1-f2)+(Iyy-Izz)*q*r)/Ixx;
  f << dot(q) == (l*(f4-f3)+(Izz-Ixx)*p*r)/Iyy;
  f << dot(r) == Ct*(f1+f2-f3-f4)/Izz;
  // Pendulum
  f << dot(pendYaw) == dPendYaw;
  f << dot(pendPitch) == dPendPitch;
  f << dot(pendRoll) == dPendRoll;
  f << dot(Wpendx) == (cos(pendRoll)*fTot*sin(pitch) - sin(pendRoll)*fTot*cos(roll)*cos(pitch))/(mq*Lpend);
  f << dot(Wpendy) ==  - ( -cos(pendPitch)*fTot*sin(roll)*cos(pitch) + sin(pendPitch)*sin(pendRoll)*fTot*sin(pitch) + sin(pendPitch)*cos(pendRoll)*fTot*cos(roll)*cos(pitch) ) / (mq*Lpend);

  ocp.subjectTo( f );

  //FIXED CONSTRAINTS:
  //Start
  ocp.subjectTo( AT_START,  yaw  ==  0. );
  ocp.subjectTo( AT_START,  dyaw ==  0. );

  //End
  ocp.subjectTo( AT_END,  yaw  ==  0. );
  ocp.subjectTo( AT_END,  dyaw ==  0. );

  //Controls
  ocp.subjectTo( 0.  <= f1 <=  umax1   );
  ocp.subjectTo( 0.  <= f2 <=  umax2   );
  ocp.subjectTo( 0.  <= f3 <=  umax3   );
  ocp.subjectTo( 0.  <= f4 <=  umax4   );

  //Avoid Flipping roll
  ocp.subjectTo( -opts.maxAngle()  <= roll <=  opts.maxAngle()   );

  //Avoid Flipping or Singularity pitch
  if (opts.maxAngle() > opts.maxAngleSing())
  {
      //Avoid Singularity
      ocp.subjectTo( -opts.maxAngleSing()  <= pitch <=  opts.maxAngleSing()  );
  }
  else
  {
      ocp.subjectTo( -opts.maxAngle()  <= pitch <=  opts.maxAngle()   );
  }

  //Time boundaries
  ocp.subjectTo( opts.Tmin()  <= T  <= opts.Tmax()  );

  //PRM CONSTRAINTS:
  //Start
  ocp.subjectTo( AT_START,  qx  ==  opts.configInit ()[0] );
  ocp.subjectTo( AT_START,  vx  ==  opts.velInit    ()[0] );

  ocp.subjectTo( AT_START,  qy  ==  opts.configInit ()[1] );
  ocp.subjectTo( AT_START,  vy  ==  opts.velInit    ()[1] );

  ocp.subjectTo( AT_START,  qz  ==  opts.configInit ()[2] );
  ocp.subjectTo( AT_START,  vz  ==  opts.velInit    ()[2] );

//  ocp.subjectTo( AT_START,  yaw ==  opts.configInit ()[3] );
  ocp.subjectTo( AT_START,  roll ==  opts.configInit ()[3] );
  ocp.subjectTo( AT_START, pitch ==  opts.configInit ()[4] );

  ocp.subjectTo( AT_START, pendRoll ==  opts.configInit ()[5] );
  ocp.subjectTo( AT_START, pendPitch ==  opts.configInit ()[6] );

//  ocp.subjectTo( AT_START,  dyaw ==  opts.configInit ()[3] );
//  ocp.subjectTo( AT_START,dpitch ==  opts.velInit    ()[3] );
//  ocp.subjectTo( AT_START, droll ==  opts.velInit    ()[4] );

  ocp.subjectTo( AT_START,  p ==  opts.velInit    ()[3] );
  ocp.subjectTo( AT_START,  q ==  opts.velInit    ()[4] );
//  ocp.subjectTo( AT_START,  r ==  opts.velInit    ()[5] );

  ocp.subjectTo( AT_START,  dPendRoll ==  opts.velInit    ()[5] );
  ocp.subjectTo( AT_START,  dPendPitch ==  opts.velInit    ()[6] );

  //End
  ocp.subjectTo( AT_END  ,  qx  ==  opts.configFinal()[0] );
  ocp.subjectTo( AT_END  ,  vx  ==  opts.velFinal   ()[0] );

  ocp.subjectTo( AT_END  ,  qy  ==  opts.configFinal()[1] );
  ocp.subjectTo( AT_END  ,  vy  ==  opts.velFinal   ()[1] );

  ocp.subjectTo( AT_END  ,  qz  ==  opts.configFinal()[2] );
  ocp.subjectTo( AT_END  ,  vz  ==  opts.velFinal   ()[2] );

//  ocp.subjectTo( AT_END,  yaw ==  opts.configFinal ()[3] );
  ocp.subjectTo( AT_END,   roll ==  opts.configFinal ()[3] );
  ocp.subjectTo( AT_END,  pitch ==  opts.configFinal ()[4] );

  ocp.subjectTo( AT_END, pendRoll ==  opts.configFinal ()[5] );
  ocp.subjectTo( AT_END, pendPitch ==  opts.configFinal ()[6] );

//  ocp.subjectTo( AT_END,  dyaw ==  opts.configFinal ()[3] );
//  ocp.subjectTo( AT_END, dpitch ==  opts.velFinal    ()[3] );
//  ocp.subjectTo( AT_END,  droll ==  opts.velFinal    ()[4] );

  ocp.subjectTo( AT_END,  p ==  opts.velFinal    ()[3] );
  ocp.subjectTo( AT_END,  q ==  opts.velFinal    ()[4] );
//  ocp.subjectTo( AT_END,  r ==  opts.velFinal    ()[5] );

  ocp.subjectTo( AT_START,  dPendRoll ==  opts.velFinal    ()[5] );
  ocp.subjectTo( AT_START,  dPendPitch ==  opts.velFinal    ()[6] );

  //ADD OBSTACLE AVOIDANCE CONSTRAINT
  if (opts.withSphericalObstacle())
  {
      std::vector<double> obs = opts.sphericalObstacle();
      //Position Constraint
      ocp.subjectTo( pow(obs[0], 2) <= square(qx-obs[1]) + square(qy-obs[2]) + square(qz-obs[3]) );

      //Position + speed*half_time_step Constraint
      ocp.subjectTo( pow(obs[0], 2) <= square(qx+vx*T/opts.steps()-obs[1]) + square(qy+vy*T/opts.steps()-obs[2])
                                        + square(qz+vz*T/opts.steps()-obs[3]) );

  }

  //  --- SETUP SOLVER --------------------

  OptimizationAlgorithm algorithm(ocp);

  const std::vector<std::string> plotNames = {"X", "Y", "Z", "roll", "pitch"};
  std::vector<ACADO::Expression> plotExpr = {qx, qy, qz, pendRoll, pendPitch};

  setupPlots(algorithm,opts,plotExpr,plotNames);

  initControlAndStateQuadcopter(algorithm,opts);

  //Full static inital guess
//  Grid timeGrid(0.0,1.,opts.steps()+1);
//  VariablesGrid x_init(10, timeGrid);
//  x_init.setAll(0);
//  VariablesGrid u_init(4, timeGrid);
//  VariablesGrid param(1,timeGrid);
//  param.setAll(15);
//  for (int i = 0 ; i<opts.steps()+1 ; i++ ) {
//      u_init(i,0) = 6.13;
//      u_init(i,1) = 6.13;
//      u_init(i,2) = 6.13;
//      u_init(i,3) = 6.13;
//  }
//  algorithm.initializeDifferentialStates(x_init);
//  algorithm.initializeControls(u_init);
//  algorithm.initializeParameters(param);

  initHorizon(algorithm,opts);
  initAlgorithmStandardParameters(algorithm,opts);

  returnValue retval = algorithm.solve();

//  outputControlAndState(algorithm,opts);
  outputControlAndStateWithoutLagrangeTerm(algorithm,opts);
  outputParameters(algorithm,opts);

  //  --- RETURN --------------------------
  std::cout << "###### Return["<<int(retval)<<"] JobID=" << opts.jobid() << timer << std::endl;
  return (int)retval;
}


