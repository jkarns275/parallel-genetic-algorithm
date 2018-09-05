## About
This is a project for my undergrad Parallel Programming class, written in Rust. The project specification is as follows:


Assignment 1
Write a parallel genetic algorithm program for an Facilities Layout problem in which:

  - There are N stations (N at least 32) and M (M at least N) spots to place them on two-dimensional space (of any shape you like) representing a one-floor factory. (There may be unoccupied spots serving as "holes".)
  - There is a metric representing the benefit (affinity) of placing any two stations A and B near each other (to transport parts), as a function of the distance between them, as well as a maximum capacity or rate. (The exact metrics are up to you; you may use random weights.) The goal is to assign spots maximizing total affinity.
  - Each of K parallel tasks solve by (at least in part randomly) swapping station spots (possibly with holes), and occasionally exchanging parts of solutions with others. (This is the main concurrent coordination problem.) Run the program on a computer with at least 32 cores (and K at least 32). (You can develop with smaller K.)
  - The program occasionally (for example twice per second) graphically displays solutions until converged or a given number of iterations. Details are up to you. 

[Doug Lea](http://gee.cs.oswego.edu/dl)
