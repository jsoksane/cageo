/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 * \file StreamBurningCommon.cpp
 */
#include "StreamBurningCommon.h"

bool coordDistCompare(CoordDist& first, CoordDist& second) {
	if (first.d < second.d) {
		return true;
	} else if (first.d == second.d) {
		if (first.p.y < second.p.y) {
			return true;
		} else if (first.p.y == second.p.y) {
			if (first.p.x < second.p.x)
				return true;
			else
				return false;
		} else
			return false;
	} else
		return false;
}
