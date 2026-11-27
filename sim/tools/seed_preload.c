/*
 * LD_PRELOAD shim: seed glibc's random()/rand() from $UAVSIM_SEED before arducopter's main().
 *
 * ArduPilot SITL (Copter-4.7.3) draws its simulated noise from random()/rand() and never seeds
 * them: IMU noise (AP_InertialSensor_SITL), baro noise (AP_Baro_SITL), wind turbulence
 * (SIM_Aircraft::update_wind), rand_normal(). Without a seed every run gets the same sequence.
 * glibc's rand() shares random()'s state, so one srandom() covers both. Seed 1 is glibc's
 * default state, i.e. identical to running without the shim.
 *
 * Built by runner/seed.py:  gcc -shared -fPIC -O2 -o libuavsim_seed.so seed_preload.c
 */
#include <stdio.h>
#include <stdlib.h>

__attribute__((constructor)) static void uavsim_seed(void)
{
    const char *s = getenv("UAVSIM_SEED");
    if (s != NULL && *s != '\0') {
        unsigned v = (unsigned)strtoul(s, NULL, 10);
        srandom(v);
        fprintf(stderr, "uavsim_seed: srandom(%u)\n", v);
    }
}
