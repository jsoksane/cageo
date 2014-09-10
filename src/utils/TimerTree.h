/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file TimerTree.h
 */
#ifndef TIMERTREE_H_
#define TIMERTREE_H_

#include "Timer.h"
#include <string>
#include <map>
#include <utility>
#include <vector>
#include <iomanip>
#include <iostream>

/**
 * \brief A class to present the timing results in a table.
 *
 * TimerTree contains one timer object, and a list to child TimerTree objects.
 */
class TimerTree {
    private:
        /**
         * \brief The indentation of the subtimers.
         */
        int indentation;
        /**
         * \brief The width of the time info box.
         */
        int widthTime;
        /**
         * \brief The width of the percent info box.
         */
        int widthPercent;
        /**
         * \brief The timer object of this node.
         */
        Timer* timer;
        /**
         * \brief The subtree that contain the subtimers.
         */
        std::vector<TimerTree*> children;

        /**
         * \brief Print the summary of the TimerTree that is child of another
         * TimerTree.
         */
        void summary(int level,
                     float total,
                     int nameWidth,
                     int depth);
    public:
        /**
         * \brief Construct an empty TimerTree.
         */
        TimerTree();
        /**
         * \brief Construct a TimerTree with \a timer Timer.
         */
        TimerTree(Timer *timer);
        /**
         * \brief Destructor.
         */
        virtual ~TimerTree() { };
        /**
         * \brief Set the default values for the printing.
         */
        void setDefaultValues();
        /**
         * \brief Set \a timer to TimerTree.
         */
        void setTimer(Timer *timer);
        /**
         * \brief Set \a timer into child TimerTree.
         */
        void setChild(Timer *timer);
        /**
         * \brief Return true, if this TimerTree object has children.
         */
        bool hasChildren();
        /**
         * \brief Return the length of the longest \a timingTarget name among
         * this and child objects.
         */
        int getMaxNameWidth();
        /**
         * \brief If this TimerTree has no children, return 0. Otherwise,
         * return 1 + maximum of getMaxDepth() among the children.
         */
        int getMaxDepth();
        /**
         * \brief Print the summary.
         */
         void summary();
};

#endif /* TIMER_H_ */
