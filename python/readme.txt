1. Run actorcritic.py, with 'RESTORE=""' . The script trains the actor-critic nets, and generate some plots to display the Q-value. At the end of the script, the policy should be able to bring the pendulum to desired position. Finally store the optimized weights with "tf.train.Saver().save(sess,'netvalues/actorcritic')"


2. Sample the optimal policy with warmstart.py. The script uses discrete_pendulum.cpp and the actor-critic nets to warm start the search. Set up RESTORE to the actor-critic weight file "RESTORE = 'netvalues/actorcritic'". Run the sampling with "D = explore(20000)". Finally save the dataset with "np.save(open('databasexx.np','w'),D)"

3. Train the policy net from the dataset with policy.py. Set "RESTORE=''" to prevent loading previously optimized weights. The net was trained for 1e6 sessions, which might be excessive. At the end of the training, the net is able to control the pendulum to desired position.

4. TODO: use the policy net to warm start the acado search. We might expect that few iterations are now necessary to get the true optimum (compared to warmstarting with the AC net).


-----

DOUBLE Pendulum

actor-critic has been modified. It produces weights in netvalues/double. These weights can be red using plot_double.py. However, the system fail to converge (even if reasonable trajectories can be obtained from the neural net).

