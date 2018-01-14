/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// set the number of particles
	num_particles = 100;
	particles.resize(num_particles);
	weights.resize(num_particles);

	default_random_engine gen;
	// This line creates a normal (Gaussian) distribution for x, y, theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	// init particle
	for (int i = 0; i < num_particles; i++) {
		particles[i].id = i;
		particles[i].weight = 1.0; 
		weights[i] = 1.0;
		// add noise
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}    
	is_initialized = true; 
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// define normal distributions for sensor noise
	default_random_engine gen;

	for (int i = 0; i < num_particles; i++) {
		Particle &p = particles[i];
		if (fabs(yaw_rate) > 0.0001) {
			p.x = p.x + velocity / yaw_rate * ( sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y = p.y + velocity / yaw_rate * (-cos(p.theta + yaw_rate * delta_t) + cos(p.theta));
			p.theta = p.theta + yaw_rate * delta_t; 	
		} else {
			p.x = p.x + velocity * delta_t * cos(p.theta);
			p.y = p.y + velocity * delta_t * sin(p.theta);
			p.theta = p.theta;

		}

		// This line creates a normal (Gaussian) distribution for x, y, theta
		normal_distribution<double> dist_x(p.x, std_pos[0]);
		normal_distribution<double> dist_y(p.y, std_pos[1]);
		normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
	}
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
  //   implement this method and use it as a helper during the updateWeights phase.

  for (unsigned int i = 0; i < observations.size(); i++) {
    
    // grab current observation
    LandmarkObs o = observations[i];

    // init minimum distance to maximum possible
    double min_distance = numeric_limits<double>::max();

    // init id of landmark from map placeholder to be associated with the observation
    int map_id = -1;
    
    for (int j=0; j < predicted.size(); j++) {
      // grab current prediction
      LandmarkObs p = predicted[j];
      
      // get distance between current/predicted landmarks
      double cur_distance = dist(o.x, o.y, p.x, p.y);

      // find the predicted landmark nearest the current observed landmark
      if (cur_distance < min_distance) {
        min_distance = cur_distance;
        map_id = p.id;
      }
    }

    // set the observation's id to the nearest predicted landmark's id
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  	for (int i=0; i < num_particles; i++) {
		  Particle &p = particles[i];

    	// create vector for map landmark locations predicted 
    	std::vector<LandmarkObs> predicted;

		// predicted landmarks within sensor range
    	for (int j=0; j < map_landmarks.landmark_list.size(); j++) {
			int 	m_id = map_landmarks.landmark_list[j].id_i;
			double 	m_x  = map_landmarks.landmark_list[j].x_f;
			double 	m_y  = map_landmarks.landmark_list[j].y_f;
      		
      		if (dist(m_x, m_y, p.x, p.y) <= sensor_range) {
	        	// add prediction to vector
    	    	predicted.push_back(LandmarkObs({m_id, m_x, m_y}));
			}
		}

		// transform the list of observations from vehicle coordinates to map coordinates
		std::vector<LandmarkObs> trans_obs;
		for (int j=0; j < observations.size(); ++j) {
			double t_x = cos(p.theta)*observations[j].x - sin(p.theta)*observations[j].y + p.x;
			double t_y = sin(p.theta)*observations[j].x + cos(p.theta)*observations[j].y + p.y;
			trans_obs.push_back(LandmarkObs({observations[j].id, t_x, t_y }));
		}

    	// perform dataAssociation for the predictions and transformed observations on current particle
    	dataAssociation(predicted, trans_obs);

    	// update weight
    	p.weight = 1.0;
	    for (int j=0; j < trans_obs.size(); j++) {
      	// placeholders for observation and associated prediction coordinates
			double o_x, o_y, pr_x, pr_y;
			o_x = trans_obs[j].x;
			o_y = trans_obs[j].y;
			int associated_prediction = trans_obs[j].id;

			// get the x,y coordinates of the prediction associated with the current observation
			for (unsigned int k = 0; k < predicted.size(); k++) {
				if (predicted[k].id == associated_prediction) {
					pr_x = predicted[k].x;
					pr_y = predicted[k].y;
				}
		}

      // calculate weight for this observation with multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double w = ( 1/ (2 * M_PI * s_x * s_y)) * exp( -( pow(pr_x - o_x, 2)
	             /(2 * pow(s_x, 2)) + (pow(pr_y - o_y, 2)/(2 * pow(s_y, 2)))));

      // product of this obersvation weight with total observations weight
      p.weight *= w;
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  
 	std::default_random_engine gen;
	// get all of the current weights
	vector<double> weights;
	for (int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
	}

	std::discrete_distribution<int> dist(weights.begin(), weights.end());
	std::vector<Particle> new_particles;

	for (int i = 0; i < particles.size(); i++) {
		int sample_index = dist(gen);
		new_particles.push_back(particles[sample_index]);
	}

	particles = new_particles;
}


Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;

}



string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}