#ifndef __PYCADO_UTILS__
#define __PYCADO_UTILS__

#include <acado_gnuplot.hpp>
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace po = boost::program_options;

struct OptionsOCP
{
  po::variables_map vm;
  ACADO::VariablesGrid Us,Xs;
  int NQ, NV, NU;
  po::options_description desc;

  

  OptionsOCP()
    : vm(),Us(),Xs(),NQ(-1),NV(-1),NU(-1),desc("Allowed options")
  {}

  virtual void addExtraOptions() {}

  void parse(int argc, const char ** argv )
  {
    desc.add_options()
      ("help,h",     "produce help message")
      ("plot",       "with plots")
      ("ostate,s",   po::value<std::string>(),                                    "Output states"  )
      ("oparam",     po::value<std::string>(),                                    "Output parameters"  )
      ("ocontrol,c", po::value<std::string>(),                                    "Output controls")
      ("icontrol,i", po::value<std::string>(),                                    "Input controls (guess)")
      ("istate,j",   po::value<std::string>(),                                    "Input states (guess)")
      ("horizon,T",  po::value<double     >()->default_value(1.0),                "Horizon length")
      ("Tmin",       po::value<double     >()->default_value(0.1),                "Horizon length minimal")
      ("Tmax",       po::value<double     >()->default_value(5.0),                "Horizon length maximal")
      ("friction",   po::value<std::vector<double> >()->multitoken()
       ->default_value(std::vector<double>{0.,0.}, "0. 0."),                      "Friction coeff")
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
      ("umin",       po::value<std::vector<double> >()->multitoken()
       ->default_value(std::vector<double>{-10.,-10.}, "-10 -10"),                 "Lower torque limit")
      ("shift,t",    po::value<int>()->default_value(0),                           "Number of time shifts")
      ("armature",   po::value<double>()->default_value(0.),                       "Joint armature")
      ("jobid",      po::value<std::string>()->default_value(""),                  "Job id, to be added to filename")
      ("statefromfile,f", "Init state from file")
      ;

    addExtraOptions();

    po::store(po::parse_command_line(argc, argv, desc,
                                     po::command_line_style::unix_style 
                                     ^ po::command_line_style::allow_short), vm);
    po::notify(vm);    
    
    if (vm.count("help")) 
      {
        std::cout << desc << "\n";
        exit(0);
      }
  }

  const double & T()        { return vm["horizon"   ].as<double>(); }
  const double & Tmin()     { return vm["Tmin"      ].as<double>(); }
  const double & Tmax()     { return vm["Tmax"      ].as<double>(); }
  const double & decay()    { return vm["decay"     ].as<double>(); }
  const int & steps()       { return vm["steps"     ].as<int>   (); }
  const int & iter()        { return vm["iter"      ].as<int>   (); }
  const int & shift()       { return vm["shift"     ].as<int>   (); }
  const int & printLevel()  { return vm["printlevel"].as<int>   (); }

  const bool   withPlot()   { return vm.count("plot")!=0; }
  const double & armature() { return vm["armature"].as<double>(); }

  const std::string & jobid()           { return vm["jobid"   ].as<std::string>(); }
  const std::string guessControlFile()  { return vm["icontrol"].as<std::string>() + jobid(); }
  const std::string guessStateFile()    { return vm["istate"  ].as<std::string>() + jobid(); }
  const std::string outputControlFile() { return vm["ocontrol"].as<std::string>() + jobid(); }
  const std::string outputStateFile()   { return vm["ostate"  ].as<std::string>() + jobid(); }
  const std::string outputParamFile()   { return vm["oparam"  ].as<std::string>() + jobid(); }

  const std::vector<double> umax()      { return vm["umax"].as< std::vector<double> >(); }
  const std::vector<double> umin()      { return vm["umin"].as< std::vector<double> >(); }
  const std::vector<double> friction()  { return vm["friction"].as< std::vector<double> >(); }

  bool withGuessControl () { return vm.count("icontrol")>0; }
  bool withGuessState   () { return vm.count("istate  ")>0; }
  bool withOutputControl() { return vm.count("ocontrol")>0; }
  bool withOutputState  () { return vm.count("ostate"  )>0; }
  bool withOutputParam  () { return vm.count("oparam"  )>0; }

  const ACADO::VariablesGrid & guessControl()
  {
    assert( guessControlFile().size()>0 );
    Us.read(guessControlFile().c_str());
    for(int loop=0;loop<shift();++loop) Us.shiftBackwards();
    return Us;
  }
  const ACADO::VariablesGrid & guessState()
  {
    assert( guessStateFile().size()>0 );
    Xs.read(guessStateFile().c_str());
    for(int loop=0;loop<shift();++loop) Xs.shiftBackwards();
    return Xs;
  }

  const std::vector<double> configInit()
  {
    std::vector<double> a_p0;
    if (vm.count("statefromfile")==0) // Init config explicit from option.
      {
        a_p0 = vm["initpos"].as< std::vector<double> >();
        if(NQ>0) a_p0.resize(NQ);
      }
    else // Init config from state file.
      {
        assert(NQ>0);
        a_p0.resize(NQ); 
        for( int loop=0;loop<NQ;++loop) 
          a_p0[loop] = Xs(0,loop);
      }
    return a_p0;
  }

  const std::vector<double> configFinal()
  {
    std::vector<double> a_p0;
    if (vm.count("statefromfile")==0) // Init config explicit from option.
      {
        a_p0 = vm["finalpos"].as< std::vector<double> >();
        if(NQ>0) a_p0.resize(NQ);
      }
    else // Init config from state file.
      {
        const uint nbp = Xs.getNumPoints();
        assert(NQ>0);
        a_p0.resize(NQ); 
        for( int loop=0;loop<NQ;++loop) 
          a_p0[loop] = Xs(nbp-1,loop);
      }
    return a_p0;
  }

  const std::vector<double> velInit()
  {
    std::vector<double> a_v0;
    if (vm.count("statefromfile")==0) // Init vel explicit from option.
      {
        a_v0 = vm["initvel"].as< std::vector<double> >();
        if(NV>0) a_v0.resize(NV);
      }
    else // Init vel from state file.
      {
        assert(NQ>0); assert(NV>0);
        a_v0.resize(NV); 
        for( int loop=0;loop<NV;++loop) 
          a_v0[loop] = Xs(0,NQ+loop);
      }
    return a_v0;
  }

  const std::vector<double> velFinal()
  {
    std::vector<double> a_v0;
    if (vm.count("statefromfile")==0) // Init vel explicit from option.
      {
        a_v0 = vm["finalvel"].as< std::vector<double> >();
        if(NV>0) a_v0.resize(NV);
      }
    else // Init vel from state file.
      {
        assert(NQ>0); assert(NV>0);
        const uint nbp = Xs.getNumPoints();
        a_v0.resize(NV); 
        for( int loop=0;loop<NV;++loop) 
          a_v0[loop] = Xs(nbp-1,NQ+loop);
      }
    return a_v0;
  }

  void displayBoundaryConditions( std::ostream& os )
  {
    os << "Init config = ";
    for( int loop=0;loop<configInit().size();++loop) os << configInit()[loop] << " ";
    os << "\nInit vel = ";
    for( int loop=0;loop<velInit().size();++loop) os << velInit()[loop] << " ";
    os << "\nFinal  config = ";
    for( int loop=0;loop<configFinal ().size();++loop) os << configFinal ()[loop] << " ";
    os << "\nFinal  vel = ";
    for( int loop=0;loop<velFinal ().size();++loop) os << velFinal ()[loop] << " ";
    os << std::endl;
  }
    

};

void initControlAndState( ACADO::OptimizationAlgorithm  & algorithm, OptionsOCP & opts )
{
  if(opts.withGuessControl()) {  algorithm.initializeControls          (opts.guessControl()); }
  if(opts.withGuessState  ()) {  algorithm.initializeDifferentialStates(opts.guessState()  ); }
}
void initHorizon( ACADO::OptimizationAlgorithm  & algorithm, OptionsOCP & opts )
{
  USING_NAMESPACE_ACADO;
  Grid timeGrid( 0.0, 1.0, opts.steps() );
  VariablesGrid   Ps( 1, timeGrid );
  Ps(0,0) = opts.T();
  algorithm.initializeParameters(Ps);
}

void initAlgorithmStandardParameters( ACADO::OptimizationAlgorithm  & algorithm, OptionsOCP & opts )
{
  USING_NAMESPACE_ACADO;
  algorithm.set( PRINTLEVEL, opts.printLevel());
  algorithm.set( PRINT_COPYRIGHT, 0);
  algorithm.set( INTEGRATOR_TYPE,INT_RK45);
  algorithm.set( MAX_NUM_ITERATIONS, opts.iter() );
}

void setupPlots( ACADO::OptimizationAlgorithm  & algorithm, OptionsOCP & opts,
                 ACADO::Expression & v0, ACADO::Expression & v1, 
                 ACADO::Expression & v2, ACADO::Expression & v3,
                 const std::string label0 = "Angle q0",
                 const std::string label1 = "Angle q1",
                 const std::string label2 = "Control u0",
                 const std::string label3 = "Control u1")
{
  USING_NAMESPACE_ACADO;
  if( opts.withPlot() )
    {
      GnuplotWindow window;
      window.addSubplot( v0, label0.c_str() );
      window.addSubplot( v1, label1.c_str() );
      window.addSubplot( v2, label2.c_str() );
      window.addSubplot( v3, label3.c_str() );
      algorithm << window;
    }
}
void outputControlAndState( ACADO::OptimizationAlgorithm  & algorithm, OptionsOCP & opts )
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
      VariablesGrid Xs;
      algorithm.getDifferentialStates( Xs );
      std::ofstream of(opts.outputStateFile().c_str());
      Xs.print( of,"","","\n",10,10,"\t","\n");
    }
}
void outputParameters( ACADO::OptimizationAlgorithm  & algorithm, OptionsOCP & opts )
{
  USING_NAMESPACE_ACADO;
  if( opts.withOutputParam() )
    {
      VariablesGrid Ps;
      algorithm.getParameters( Ps );
      std::ofstream of(opts.outputParamFile().c_str());
      Ps.print( of,"","","\n",10,10,"\t","\n");
    }
}

struct Timer
{
  boost::posix_time::ptime tstart;
  Timer() 
  {
    tstart = boost::posix_time::microsec_clock::local_time();
  }
  friend std::ostream & operator<< (std::ostream & os, const Timer & self)
  {
    boost::posix_time::ptime tend = boost::posix_time::microsec_clock::local_time();
    os << "  T=" << self.tstart
       << "  - t=" << (tend-self.tstart).total_milliseconds();
    return os;
  }
};


#endif // __PYCADO_UTILS__
