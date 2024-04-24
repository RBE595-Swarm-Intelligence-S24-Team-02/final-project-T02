#ifndef MCCA_H
#define MCCA_H

#include <buzz/argos/buzz_loop_functions.h>
#include <argos3/core/utility/math/rng.h>
#include <argos3/plugins/robots/kheperaiv/simulator/kheperaiv_entity.h>

class CMCCA : public CBuzzLoopFunctions {

public:

   CMCCA() {}
   virtual ~CMCCA() {}

   /**
    * Executes user-defined initialization logic.
    * @param t_tree The 'loop_functions' XML configuration tree.
    */
   virtual void Init(TConfigurationNode& t_tree);

   /**
    * Executes user-defined reset logic.
    * This method should restore the state of the simulation at it was right
    * after Init() was called.
    * @see Init()
    */
   virtual void Reset();

   /**
    * Executes user-defined logic right after a control step is executed.
    */
   virtual void PostStep();
   
   /**
    * Returns true if the experiment is finished, false otherwise.
    *
    * This method allows the user to specify experiment-specific ending
    * conditions. If this function returns false and a time limit is set in the
    * .argos file, the experiment will reach the time limit and end there. If no
    * time limit was set, then this function is the only ending condition.
    *
    * @return true if the experiment is finished.
    */
   virtual bool IsExperimentFinished();

   /**
    * Executes user-defined destruction logic.
    * This method should undo whatever is done in Init().
    * @see Init()
    */
   virtual void Destroy();

   virtual void BuzzBytecodeUpdated();

private:

   int GetNumRobots() const;

private:

   const float TIMESTEP = 0.1;

   CSpace::TMapPerType robot_vector;

   /** The stimuli associated to the tasks */
   std::vector<Real> m_vecPosX;

   /** The stimuli associated to the tasks */
   std::vector<Real> m_vecPosY;

   /** The stimuli associated to the tasks */
   std::vector<Real> m_vecPrevPosX;

   /** The stimuli associated to the tasks */
   std::vector<Real> m_vecPrevPosY;

   /** The stimuli associated to the tasks */
   std::vector<Real> m_vecVelX;

   /** The stimuli associated to the tasks */
   std::vector<Real> m_vecVelY;

   /** The travel distance for each robot */
   std::vector<Real> m_vecTravelDistances;

   /** The travel time for each robot */
   std::vector<Real> m_vecTravelTimes;

   /** The success condition for each robot */
   std::vector<Real> m_vecSuccess;

   /** The current position of each robot */
   std::vector<Real> m_vecPositionsX;

   /** The current position of each robot */
   std::vector<Real> m_vecPositionsY;

   /** The output file name */
   std::string m_strOutFile;

   /** The output file stream */
   std::ofstream m_cOutFile;
};

#endif
