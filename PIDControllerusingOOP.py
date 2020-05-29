"""Implementation of PID Controller using Classes.

When prompted for the value of the proportional, integral derivative gain, it was found that Kp = 0.1, Ki = 70, Kd = 0 worked
well for the problem statement.

Efforts were made to make use of Single Responsibility Principle and Object Encapsulation.
  
  Typical usage example:

  #Create instance of PID Object; User will be prompted to input proportional, integral and derivative gain
  PIDInstance1 = PID()
  
  #Create instance of PID_Calculator from the instance of the PID object; This is to inherit the instance variables
  PIDInstance1_Calculator = PID_Calculator(PIDInstance1)
  
  #Call the Calculate Controller Output method so that the arrays of values we need for plotting are calculated and created
  PIDInstance1_Calculator.Calculate_Controller_Output()
  
  #Create instance of Plot_PID
  PIDInstance1_graph = Plot_PID(PIDInstance1_Calculator.u_plot, PIDInstance1_Calculator.r_plot, PIDInstance1_Calculator.t_plot)
  
  #Plot the graph
  PIDInstance1_graph.Plot()
"""

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from __future__ import division  #to facilitate integer division


###Single Responsibility Principle
class PID(object):
    """PID solely meant to hold all the values that define the problem statement
    
    Set_Problem_Statement allows the user to changes the values of the protected members.
    Get_Problem_Statement allows the user to print out the values of the protected members.
    
    This assumes the reference signal is a single step input.
    
    Attributes:
        Kp: Proportional gain of PID instance
        Ki: Integral gain of PID Instance
        Kd: Derivative gain of PID Instance
        
        Protected members:
        u: Initial process variable value
        ri: Initial reference signal value
        rf: Final reference signal value
        t_change_in_reference_signal: Time when the reference signal changes in value
        t_total: Total time span

    """
    
    ##predefine the problem statement variables as attributes
    u = 0
    ri = 0
    rf = 5
    t_change_in_reference_signal = 2
    t_total = 5
    
    def __init__(self):
        """Inits all attributes for the instance"""
        #Kp, Ki and Kd are gotten from user input because task requires us to type it into the command line
        self.Kp = input("Proportional Gain Kp: ") #Use Kp = 0.1
        self.Ki = input("Proportional Gain Ki: ") #Use Ki = 70
        self.Kd = input("Proportional Gain Kd: ") #Use Kd = 0
        ###Object Encapsulation
        #make the problem statement variables protected members: still accessible 
        self._u = PID.u
        self._ri = PID.ri
        self._rf = PID.rf
        self._t_change_in_reference_signal = PID.t_change_in_reference_signal
        self._t_total = PID.t_total
        
    def Set_Problem_Statement(self, u, ri, rf, t_change_in_reference_signal, t_total):
        """After creating an instance, we can set attributes of that instance to the specifications in the problem statement"""
        self._u = u
        self._ri = ri
        self._rf = rf
        self._t_change_in_reference_signal = t_change_in_reference_signal
        self._t_total = t_total
    
    def Get_Problem_Statement(self):
        """Prints out the values that define the problem statement"""
        print("Problem Statement Variables:")
        print("Initial Process variable Value: " + str(self._u))
        print("Initial Reference signal: " + str(self._ri))
        print("Final reference signal: " + str(self._rf))
        print("Time when reference signal changes: " + str(self._t_change_in_reference_signal))
        print("Total time span: " + str(self._t_total))


class PID_Calculator(PID):
    """PID_Calculator solely meant to calculate the controller output at each instance of time
    
    PID_Calculator takes in an instance of a PID object and calculates the controller output for that object
    Reasoning for this being that would calculate the controller output only for each instance of a PID object, because
    Each instance is supposed to contain unique values which will give unique controller output.
    
    Set_Number_Of_Steps allows the user to change the number of steps the calculator is supposed to iterate through which
    is a private member.
    Get_Number_Of_Steps allows the user to print out the number of steps the calculator is supposed to iterate through.
    
    This assumes the reference signal is a single step input.
    
    Attributes:
        Kp: Proportional gain of PID instance
        Ki: Integral gain of PID Instance
        Kd: Derivative gain of PID Instance
        
        Protected members:
        u: Initial process variable value
        ri: Initial reference signal value
        rf: Final reference signal value
        t_change_in_reference_signal: Time when the reference signal changes in value
        t_total: Total time span
        
        Private members:
        ns: Number of steps the calculator will iterate through

    """
    
    #Number of steps the Calculator will iterate through
    ns = 500
    
    def __init__(self, PID_object):
        """Inherit variables of the instance of the PID object"""
        self.Kp = PID_object.Kp
        self.Ki = PID_object.Ki
        self.Kd = PID_object.Kd
        self._u = PID_object._u
        self._ri = PID_object.ri
        self._rf = PID_object.rf
        self._t_change_in_reference_signal = PID_object.t_change_in_reference_signal
        self._t_total = PID_object.t_total
        ###Object Encapsulation
        #make the number of steps a private member so that it cannot be accessed by any instance or outside any class
        self.__ns = PID_Calculator.ns
        
    def Set_Number_Of_Steps(self, ns):
        """Allows user to set the number of steps that the calculator will iterate through; necessary because ns is a 
        private member"""
        self.__ns = ns
        
    def Get_Number_Of_Steps(self):
        """Prints out the number of steps that the calculator will iterate through"""
        print("Number of iterations calculator is using: " + str(self.__ns))
        
    def Calculate_Controller_Output(self):
        """Calculates out the array of values for the process variable"""
        #Create empty arrays to be populated
        u = np.zeros(self.ns+1)  # array for process variable (which also happens to be the controller output in this case)
        e = np.zeros(self.ns+1)   # array for error term
        ie = np.zeros(self.ns+1)  # array for integral term
        dpv = np.zeros(self.ns+1) # array for derivative of process variable
        P = np.zeros(self.ns+1)   # array for proportional term
        I = np.zeros(self.ns+1)   # array for integral term
        D = np.zeros(self.ns+1)   # array for derivative term
        r = np.zeros(self.ns+1)  # array for reference signal
        
        #Set up array for reference signal
        r[:] = self.ri
        r[int((self.t_change_in_reference_signal)/((self.t_total)/(self.ns))):] = self.rf
        
        #Set up array for time
        t = np.linspace(0, self.t_total, self.ns + 1)
        delta_t = t[1]-t[0]
        
        #Calculate Controller output at each iteration
        for i in range(0,self.ns):
            e[i] = r[i] - u[i]
            if i >= 1:  # calculate starting on second cycle
                dpv[i] = -(u[i]-u[i-1])/delta_t
                ie[i] = ie[i-1] + e[i] * delta_t
            P[i] = self.Kp * e[i]
            I[i] = self.Ki * ie[i]
            D[i] = self.Kd * dpv[i]
            #Calculate controller output, which also happens to be the process variable
            u[i+1] = P[i] + I[i] + D[i]
        #Create new variables u_plot, r_plot, t_plot
        self.u_plot = u
        self.r_plot = r
        self.t_plot = t
        

class Plot_PID:
    """Plot_PID solely meant to plot out a graph of the controller output with respect to time against the graph of 
    reference signal with respect to time

    Plot_PID uses the array of process variable values, the array of reference signal values and the array of time values to 
    plot the graph of desired thruster speed and actual thruster speed
    
    This assumes the reference signal is a single step input.
    
    Attributes:
        u: Array of process variable values
        r: Array of reference signal values
        t: Array of time values

    """
    def __init__(self, u, r, t):
        """Inits the attributes for the arrays of values for the process variable, reference signal and time"""
        self.u = u
        self.r = r
        self.t = t
        
    def Plot(self):
        """Plots out the graphs"""
        plt.plot(self.t,self.r,'r',linewidth=2)
        plt.plot(self.t,self.u,'b--',linewidth=3)
        plt.legend(['Desired Thruster Speed','Actual Thruster Speed'],loc='best')
        plt.ylabel('Process')
        plt.ylim([-0.1,6])
        plt.xlabel('Time')
        plt.show()
        
        
#Create instance of PID Object; User will be prompted to input proportional, integral and derivative gain
PIDInstance1 = PID()
#Create instance of PID_Calculator from the instance of the PID object; This is to inherit the instance variables
PIDInstance1_Calculator = PID_Calculator(PIDInstance1)
#Call the Calculate Controller Output method so that the arrays of values we need for plotting are calculated and created
PIDInstance1_Calculator.Calculate_Controller_Output()
#Create instance of Plot_PID
PIDInstance1_graph = Plot_PID(PIDInstance1_Calculator.u_plot, PIDInstance1_Calculator.r_plot, PIDInstance1_Calculator.t_plot)
#Plot the graph
PIDInstance1_graph.Plot()

