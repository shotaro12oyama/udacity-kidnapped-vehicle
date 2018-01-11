/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
#include <math.h> 
#include <sstream>
#include <string>
#include <iterator>
#include <map>

#include "particle_filter.h"

using namespace std;




void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	default_random_engine gen;
	// This line creates a normal (Gaussian) distribution for x, y, theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	for (int i = 0; i < num_particles; ++i) {
		Particle sample_p;
		sample_p.x = dist_x(gen);
		sample_p.y = dist_y(gen);
		sample_p.theta = dist_theta(gen);
		sample_p.weight = 1.0; 
		particles.push_back(sample_p);	
	}    
	is_initialized = true; 
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	
	for (int i = 0; i < num_particles; ++i) {
		particles[i].x = particles[i].x + ((velocity / yaw_rate) *  sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
		particles[i].y = particles[i].y + ((velocity / yaw_rate) * -cos(particles[i].theta + yaw_rate * delta_t) + sin(particles[i].theta));
		particles[i].theta = particles[i].theta + yaw_rate * delta_t; 	

		default_random_engine gen;	
		// This line creates a normal (Gaussian) distribution for x, y, theta
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}    
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); ++i) {
		for(int j = 0; j < predicted.size(); ++j) {
			double min_distance = sensor_range;
			double tmp_distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if(min_distance > tmp_distance){
				min_distance = tmp_distance;
				predicted[j].id = observations[i].id;
			}
		}
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

	std::vector<LandmarkObs> landmark_observation;
	std::vector<LandmarkObs> predicted;
	for (int i=0; i<num_particles; ++i){
		double weight;
		for (int j=0; j< observations.size(); ++j) {
			LandmarkObs obsmap;
			obsmap.x= particles[i].x + (cos(particles[i].theta) * observations[j].x) - (sin(particles[i].theta) * observations[j].y);
			obsmap.y= particles[i].y + (sin(particles[i].theta) * observations[j].x) + (cos(particles[i].theta) * observations[j].y);
			predicted.push_back(obsmap);
		}

		for (int k=0; k< map_landmarks.landmark_list.size(); ++k) {
			LandmarkObs map_landmark;
			map_landmark.id = map_landmarks.landmark_list[k].id_i;
			map_landmark.x = map_landmarks.landmark_list[k].x_f;
			map_landmark.y = map_landmarks.landmark_list[k].y_f;
			landmark_observation.push_back(map_landmark);
		}

		for (int i = 0; i < landmark_observation.size(); ++i) {
			double min_distance;
			double tmp_distance;
			double mu_x;
			double mu_y;
			double p_x;
			double p_y;
			for(int j = 0; j < predicted.size(); ++j) {
				min_distance = sensor_range;
				tmp_distance = dist(predicted[j].x, predicted[j].y, landmark_observation[i].x, landmark_observation[i].y);
				if(min_distance > tmp_distance){
					min_distance = tmp_distance;
					predicted[j].id = landmark_observation[i].id;
					mu_x = landmark_observation[i].x;
					mu_y = landmark_observation[i].y;
					p_x = predicted[i].x;
					p_y = predicted[i].y;
				}
			}

			//calculate normalization term
			double gauss_norm = (1/(2 * M_PI * std_landmark[0] * std_landmark[1]));
			// calculate exponent
			double exponent = (pow(p_x - mu_x, 2) / (2 * pow(std_landmark[0], 2)) + pow(p_y - mu_y, 2) / (2 * pow(std_landmark[1], 2)));
			// calculate weight using normalization terms and exponent
			weight = gauss_norm * exp(-exponent);
			particles[i].weight *= weight;
		}
		weights.push_back(weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	for (int i=0; i<num_particles; ++i) {
		weights.push_back(particles[i].weight);
	}

	std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());
    std::vector<Particle> particles_new;
    for(int n=0; n<num_particles; ++n) {
        particles_new.push_back(particles[d(gen)]);
    }
	particles = particles_new;
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
