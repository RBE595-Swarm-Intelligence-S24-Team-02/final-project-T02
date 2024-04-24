#include "MCCA.h"
#include "buzz/buzzvm.h"

std::string robot_ids[20] = {"kiv0", "kiv1", "kiv2", "kiv3", "kiv4", "kiv5", "kiv6", "kiv7", "kiv8", "kiv9", "kiv10", "kiv11", "kiv12", "kiv13", "kiv14", "kiv15", "kiv16", "kiv17", "kiv18", "kiv19"};

/****************************************/
/****************************************/

/**
 * Functor to get data from the robots
 */
struct GetRobotData : public CBuzzLoopFunctions::COperation {

   /** Constructor */
   GetRobotData() {}

   /** The action happens here */
   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm) {

      /* Get the travel distance */
      buzzobj_t tTravelDistance = BuzzGet(t_vm, "travel_distance");
      /* Make sure it's the type we expect (a float) */
      if(!buzzobj_isfloat(tTravelDistance)) {
         LOGERR << str_robot_id << ": variable 'travel_distance' has wrong type " << buzztype_desc[tTravelDistance->o.type] << std::endl;
         return;
      }
      /* Get the value */
      float fTravelDistance = buzzobj_getfloat(tTravelDistance);
      /* Make sure the value is reasonable */
      if(fTravelDistance < 0.0) {
         LOGERR << str_robot_id << ": variable 'travel_distance' has negative value" << fTravelDistance<< std::endl;
         return;
      }
      /* Set the mapping */
      m_vecTravelDistances[t_vm->robot] = fTravelDistance;

      /* Get the travel time*/
      buzzobj_t tTravelTime = BuzzGet(t_vm, "travel_time");
      /* Make sure it's the type we expect (a float) */
      if(!buzzobj_isfloat(tTravelTime)) {
         LOGERR << str_robot_id << ": variable 'travel_time' has wrong type " << buzztype_desc[tTravelTime->o.type] << std::endl;
         return;
      }
      /* Get the value */
      float fTravelTime = buzzobj_getfloat(tTravelTime);
      /* Make sure the value is reasonable */
      if(fTravelTime < 0.0) {
         LOGERR << str_robot_id << ": variable 'travel_time' has negative value" << fTravelTime<< std::endl;
         return;
      }
      /* Set the mapping */
      m_vecTravelTimes[t_vm->robot] = fTravelTime;

      /* Get success criterion */
      buzzobj_t tSuccess = BuzzGet(t_vm, "success");
      /* Make sure it's the type we expect (an int) */
      if(!buzzobj_isint(tSuccess)) {
         LOGERR << str_robot_id << ": variable 'success' has wrong type " << buzztype_desc[tSuccess->o.type] << std::endl;
         return;
      }
      /* Get the value */
      int nSuccess = buzzobj_getfloat(tSuccess);
      /* Make sure the value is reasonable */
      if(nSuccess < 0 || nSuccess > 1) {
         LOGERR << str_robot_id << ": variable 'success' has incorrect value" << nSuccess<< std::endl;
         return;
      }
      /* Set the mapping */
      m_vecSuccess[t_vm->robot] = nSuccess;
   }

   /** Travel distances */
   std::map<int,Real> m_vecTravelDistances;
   /** Travel times*/
   std::map<int,Real> m_vecTravelTimes;
   /** Success */
   std::map<int,int> m_vecSuccess;
};

/****************************************/
/****************************************/

/* Insert posX table into Buzz VMs */
struct PutPosX : public CBuzzLoopFunctions::COperation {

   PutPosX(const std::vector<Real>& vec_posX) : m_vecPosX(vec_posX) {}
   
   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm) {
      BuzzTableOpen(t_vm, "posX");
      for(int i = 0; i < m_vecPosX.size(); ++i) {
         BuzzTablePut(t_vm, i, static_cast<float>(m_vecPosX[i]));
      }
      BuzzTableClose(t_vm);
   }
   const std::vector<Real>& m_vecPosX;
};

/* Insert posY table into Buzz VMs */
struct PutPosY : public CBuzzLoopFunctions::COperation {

   PutPosY(const std::vector<Real>& vec_posY) : m_vecPosY(vec_posY) {}
   
   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm) {
      BuzzTableOpen(t_vm, "posY");
      for(int i = 0; i < m_vecPosY.size(); ++i) {
         BuzzTablePut(t_vm, i, static_cast<float>(m_vecPosY[i]));
      }
      BuzzTableClose(t_vm);
   }
   const std::vector<Real>& m_vecPosY;
};

/* Insert velX table into Buzz VMs */
struct PutVelX : public CBuzzLoopFunctions::COperation {

   PutVelX(const std::vector<Real>& vec_velX) : m_vecVelX(vec_velX) {}
   
   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm) {
      BuzzTableOpen(t_vm, "velX");
      for(int i = 0; i < m_vecVelX.size(); ++i) {
         BuzzTablePut(t_vm, i, static_cast<float>(m_vecVelX[i]));
      }
      BuzzTableClose(t_vm);
   }
   const std::vector<Real>& m_vecVelX;
};

/* Insert velY table into Buzz VMs */
struct PutVelY : public CBuzzLoopFunctions::COperation {

   PutVelY(const std::vector<Real>& vec_velY) : m_vecVelY(vec_velY) {}
   
   virtual void operator()(const std::string& str_robot_id,
                           buzzvm_t t_vm) {
      BuzzTableOpen(t_vm, "velY");
      for(int i = 0; i < m_vecVelY.size(); ++i) {
         BuzzTablePut(t_vm, i, static_cast<float>(m_vecVelY[i]));
      }
      BuzzTableClose(t_vm);
   }
   const std::vector<Real>& m_vecVelY;
};

/****************************************/
/****************************************/

void CMCCA::Init(TConfigurationNode& t_tree) {
   /* Call parent Init() */
   CBuzzLoopFunctions::Init(t_tree);
   /* Parse XML tree */
   GetNodeAttribute(t_tree, "outfile", m_strOutFile);

   /* Resize position/velocity vectors */
   m_vecPosX.resize(20);
   m_vecPosY.resize(20);
   m_vecPrevPosX.resize(20);
   m_vecPrevPosY.resize(20);
   m_vecVelX.resize(20);
   m_vecVelY.resize(20);

   /* Initialize the robot positions and velocities */
   m_vecPositionsX.reserve(20);
   m_vecPositionsY.reserve(20);

   /* Get robot position data */
   robot_vector = GetSpace().GetEntitiesByType("kheperaiv");
   for(int i = 0; i < GetNumRobots(); ++i) {
      CKheperaIVEntity* robot = any_cast<CKheperaIVEntity*>(robot_vector[robot_ids[i]]);
      CVector3 robot_pos = robot->GetEmbodiedEntity().GetOriginAnchor().Position;
      m_vecPositionsX[i] = robot_pos.GetX();
      m_vecPositionsY[i] = robot_pos.GetY();
   }

   /* Update position and velocity vectors */
   for(int i = 0; i < GetNumRobots(); ++i) {
      m_vecPosX[i] = m_vecPositionsX[i];
      m_vecPosY[i] = m_vecPositionsY[i];
      m_vecVelX[i] = (m_vecPosX[i] - m_vecPrevPosX[i]) / TIMESTEP;
      m_vecVelY[i] = (m_vecPosY[i] - m_vecPrevPosY[i]) / TIMESTEP;
   }

   /** Convey the robot positions and velocities to every robot */
   BuzzForeachVM(PutPosX(m_vecPosX));
   BuzzForeachVM(PutPosY(m_vecPosY));
   BuzzForeachVM(PutVelX(m_vecVelX));
   BuzzForeachVM(PutVelY(m_vecVelY));

   /* Open the output file */
   m_cOutFile.open(m_strOutFile.c_str(),
                   std::ofstream::out | std::ofstream::trunc);
}

/****************************************/
/****************************************/

void CMCCA::Reset() {
   
   /* Get robot position data */
   robot_vector = GetSpace().GetEntitiesByType("kheperaiv");
   for(int i = 0; i < GetNumRobots(); ++i) {
      CKheperaIVEntity* robot = any_cast<CKheperaIVEntity*>(robot_vector[robot_ids[i]]);
      CVector3 robot_pos = robot->GetEmbodiedEntity().GetOriginAnchor().Position;
      m_vecPositionsX[i] = robot_pos.GetX();
      m_vecPositionsY[i] = robot_pos.GetY();
   }

   /* Update position and velocity vectors */
   for(int i = 0; i < GetNumRobots(); ++i) {
      m_vecPosX[i] = m_vecPositionsX[i];
      m_vecPosY[i] = m_vecPositionsY[i];
      m_vecVelX[i] = (m_vecPosX[i] - m_vecPrevPosX[i]) / TIMESTEP;
      m_vecVelY[i] = (m_vecPosY[i] - m_vecPrevPosY[i]) / TIMESTEP;
   }

   /** Convey the robot positions and velocities to every robot */
   BuzzForeachVM(PutPosX(m_vecPosX));
   BuzzForeachVM(PutPosY(m_vecPosY));
   BuzzForeachVM(PutVelX(m_vecVelX));
   BuzzForeachVM(PutVelY(m_vecVelY));

   /* Reset the output file */
   m_cOutFile.open(m_strOutFile.c_str(),
                   std::ofstream::out | std::ofstream::trunc);

}

/****************************************/
/****************************************/

void CMCCA::Destroy() {
   m_cOutFile.close();
}

/****************************************/
/****************************************/

void CMCCA::PostStep() {

   /* Get robot data */
   GetRobotData cGetRobotData;
   BuzzForeachVM(cGetRobotData);

   /* Save previous positions */
   m_vecPrevPosX = m_vecPosX;
   m_vecPrevPosY = m_vecPosY;

   /* Update robot positions/velocities */
   robot_vector = GetSpace().GetEntitiesByType("kheperaiv");
   for(int i = 0; i < GetNumRobots(); ++i) {
      CKheperaIVEntity* robot = any_cast<CKheperaIVEntity*>(robot_vector[robot_ids[i]]);
      CVector3 robot_pos = robot->GetEmbodiedEntity().GetOriginAnchor().Position;
      m_vecPositionsX[i] = robot_pos.GetX();
      m_vecPositionsY[i] = robot_pos.GetY();
   }

   /* Update position and velocity vectors */
   for(int i = 0; i < GetNumRobots(); ++i) {
      m_vecPosX[i] = m_vecPositionsX[i];
      m_vecPosY[i] = m_vecPositionsY[i];
      m_vecVelX[i] = (m_vecPosX[i] - m_vecPrevPosX[i]) / TIMESTEP;
      m_vecVelY[i] = (m_vecPosY[i] - m_vecPrevPosY[i]) / TIMESTEP;
   }

   /* Convey the robot positions and velocities to every robot */
   BuzzForeachVM(PutPosX(m_vecPosX));
   BuzzForeachVM(PutPosY(m_vecPosY));
   BuzzForeachVM(PutVelX(m_vecVelX));
   BuzzForeachVM(PutVelY(m_vecVelY));

   /* Log output data to file */
   for(int i = 0; i < GetNumRobots(); ++i) {
      m_cOutFile << GetSpace().GetSimulationClock() << "\t"
                  << i << "\t"
                  // << m_vecPositionsX[i] << "\t"
                  // << m_vecPositionsY[i] << "\t"
                  << cGetRobotData.m_vecTravelDistances[i] << "\t"
                  << cGetRobotData.m_vecTravelTimes[i] << "\t"
                  << cGetRobotData.m_vecSuccess[i];
      m_cOutFile << std::endl;
   }
}

/****************************************/
/****************************************/

bool CMCCA::IsExperimentFinished() {
   /* Feel free to try out custom ending conditions */
   return false;
}

/****************************************/
/****************************************/

int CMCCA::GetNumRobots() const {
   return m_mapBuzzVMs.size();
}

/****************************************/
/****************************************/

void CMCCA::BuzzBytecodeUpdated() {
   BuzzForeachVM(PutPosX(m_vecPosX));
   BuzzForeachVM(PutPosY(m_vecPosY));
   BuzzForeachVM(PutVelX(m_vecVelX));
   BuzzForeachVM(PutVelY(m_vecVelY));
}

/****************************************/
/****************************************/

REGISTER_LOOP_FUNCTIONS(CMCCA, "MCCA");
