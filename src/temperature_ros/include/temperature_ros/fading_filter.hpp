#ifndef FADING_FILTER_H_
#define FADING_FILTER_H_

#include <string>

namespace fading_filter{
    class FadingFilter{
        public:
            std::string id;
            double beta;
            double x;
            double dx;
            
            // Consturctor/Deconstructor
            FadingFilter(){/*do nothing*/}
            FadingFilter(double beta){  
                this->beta=beta;
                this->x = 0;
                this->dx = 0;
            }
            FadingFilter(std::string id, double beta, double x0, double dx0){
                this->id = id;
                this->beta = beta;
                this->x = x0;
                this->dx = dx0;
            }
            ~FadingFilter(){/*do nothing*/}

            // Update
            void update(double y, double dt){
                double x_old = this->x;
                double dx_old = this->dx;

                double G = 1-beta*beta;
                double H = (1-beta)*(1-beta);

                this->x = x_old + dx_old*dt + G*(y - (x_old + dx_old*dt));
                this->dx = dx_old + H/dt*(y - (x_old + dx_old*dt));
            }
    };

} // namespace fading_filter

#endif // FADING_FILTER_H_