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
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define NUMBER_OF_PARTICLES 500

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	num_particles = NUMBER_OF_PARTICLES;
	// Create normal distributions for x, y and theta.
	default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	
	//Resize particles and weights size
	particles.resize(num_particles);
	
	// Initializes particles
    for (int i = 0; i < num_particles; ++i) {

      particles[i].id = i;
      particles[i].x = dist_x(gen);
      particles[i].y = dist_y(gen);
      particles[i].theta = dist_theta(gen);
      particles[i].weight = 1.0;
    }
	
	//Initialization finished
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	std::default_random_engine gen;
	
	// This line creates a normal (Gaussian) distribution for x, y and theta
    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[2]);
	
	for (int i = 0; i < num_particles; i++) {
        //avoid division by zero
        if (fabs(yaw_rate) > 0.001) {
            particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }
        else {
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }

        particles[i].x +=  dist_x(gen);
        particles[i].y +=  dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (unsigned int i = 0; i < observations.size(); i++) {
        double min_dist = numeric_limits<double>::max();
        int min_index = -1;

        for (unsigned int j = 0; j < predicted.size(); j++) {
            double cur_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if (cur_dist < min_dist) {
                min_dist = cur_dist;
                min_index = j;
            }
        }
        // assign nearest neighbor
        observations[i].id = predicted[min_index].id;
    }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	//init weights
	weights.clear();
	// calculate normalization term
	double sigma_xx = std_landmark[0]*std_landmark[0];
	double sigma_yy = std_landmark[1]*std_landmark[1];
	double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
	
	for (int i = 0; i < num_particles; i++) {
      vector<LandmarkObs> landmarks_on_map;
	  vector<LandmarkObs> obs_on_map;
	  
	  // find possible landmarks in range
	  for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            double cur_dist = dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f,
                                   particles[j].x, particles[j].y);
            if (cur_dist <= sensor_range) {
                LandmarkObs landmark;
                landmark.id = map_landmarks.landmark_list[j].id_i;
                landmark.x = map_landmarks.landmark_list[j].x_f;
                landmark.y = map_landmarks.landmark_list[j].y_f;

                landmarks_on_map.push_back(landmark);
            }
        }
		
	    // convert observation to map's coordinate system
        for (unsigned int j = 0; j < observations.size(); j++) {
            double cur_dist = dist(observations[j].x, observations[j].y, 0, 0);
            if (cur_dist <= sensor_range) {
                LandmarkObs landmark;
                landmark.id = -1;
                landmark.x = particles[i].x + observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta);
                landmark.y = particles[i].y + observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta);

                obs_on_map.push_back(landmark);
            }
		}
		
		dataAssociation(landmarks_on_map, obs_on_map);
		
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
		
		// initial weight
        particles[i].weight = 1.0;
		
		for (unsigned int j = 0; j < obs_on_map.size(); j++){
			
			double dx, dy;
			
			for (unsigned int k = 0; k < landmarks_on_map.size(); k++) {
              if (landmarks_on_map[k].id == obs_on_map[j].id){
                dx = obs_on_map[j].x - landmarks_on_map[k].x;
                dy = obs_on_map[j].y - landmarks_on_map[k].y;
              }
			}
			
            particles[i].weight *= gauss_norm * exp(-0.5 * ((dx * dx) / (sigma_xx) + (dy * dy) / (sigma_yy)));
			
			associations.push_back(obs_on_map[j].id);
			sense_x.push_back(obs_on_map[j].x);
			sense_y.push_back(obs_on_map[j].y);
        }
		// Update weights
		weights.push_back(particles[i].weight);
		// Set particle associations for debugging
		particles[i] = SetAssociations(particles[i], associations, sense_x, sense_y);
	
	}
}


void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	default_random_engine gen;
	discrete_distribution<int> dist_particles(weights.begin(), weights.end());
	
	vector<Particle> new_particles;
    new_particles.resize(num_particles);
	
	for(int i=0; i<num_particles; ++i){
		new_particles[i] = particles[dist_particles(gen)];
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
	
    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

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
