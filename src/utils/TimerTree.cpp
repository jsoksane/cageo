/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file TimerTree.cpp
 */
#include "TimerTree.h"

typedef std::vector<TimerTree*> vec;

TimerTree::TimerTree() {
    setDefaultValues();
};

TimerTree::TimerTree(Timer *timer) {
    setTimer(timer);
    setDefaultValues();
};

void TimerTree::setDefaultValues() {
    indentation = 2;
    widthTime = 11;
    widthPercent = 12;
};

void TimerTree::setTimer(Timer *timer) {
    this->timer = timer;
};

void TimerTree::setChild(Timer *timer) {
    TimerTree *tt = new TimerTree(timer);
    children.push_back(tt);
};

bool TimerTree::hasChildren() {
    return children.size();
};

int TimerTree::getMaxNameWidth() {
    int max = timer->getTimingTarget().size();
    for (vec::iterator it = children.begin(); it != children.end(); ++it) {
        if ((*it)->getMaxNameWidth() > max) {
            max = (*it)->getMaxNameWidth();
        }
    }
    return max;
};

int TimerTree::getMaxDepth() {
    int max = 0;
    if (!hasChildren()) {
        return max;
    }
    for (vec::iterator it = children.begin(); it != children.end(); ++it) {
        if ((*it)->getMaxDepth() > max) {
            max = (*it)->getMaxDepth();
        }
    }
    return max + 1;
};

/* 
 * +---------------+--------------------------+----------------+-------------+
 * | Timer summary |               Total time |  Average time  |  % of total |
 * +---------------+------+-------------+-----+----------+-----+-------------+
 * | timer1               |        12.4 |  ms |      0.6 |  ms |        76.4 |
 * +---+------------------+-------------+-----+----------+-----+-------------+
 * |   | subtimer1        |         8.4 |  ms |      0.4 |  ms |         4.5 |
 * +---+---+--------------+-------------+-----+----------+-----+-------------+
 * |       | subsubtimer1 |         8.4 |  ms |      0.4 |  ms |         4.5 |
 * +-------+--------------+-------------+-----+----------+-----+-------------+
 */
void TimerTree::summary(int level, float total, int nameWidth, int depth) {
    // The indentation
    std::cout << std::setw(indentation * level) << "";
    // The name
    std::cout.width(nameWidth + (depth - level) * indentation);
    std::cout << std::left << timer->getTimingTarget();
    // The absolute time
    std::cout.setf(std::ios::right);
    std::cout.setf(std::ios::fixed);
    std::cout.precision(1);
    std::cout << std::setw(widthTime) << timer->getTotalTime() << " ms";
    // The average time
    std::cout << std::setw(widthTime) << timer->getAverageTime() << " ms";
    // The percentage
    std::cout << std::setw(widthPercent) << 
        (100 * timer->getTotalTime() / total) << std::endl;
    for (vec::iterator it = children.begin(); it != children.end(); ++it) {
        (*it)->summary(level + 1, timer->getTotalTime(), nameWidth, depth);
    }
};

void TimerTree::summary() {
    float total = timer->getTotalTime();
    int depth = getMaxDepth();
    int nameWidth = getMaxNameWidth();
    int B = indentation * depth + nameWidth;

    // Print the header line
    std::cout.fill(' ');
    std::cout.setf(std::ios::right);
    std::cout << "Timer summary";
    std::cout << std::setw(B + widthTime + 3 - 13) << "Total time";
    std::cout << std::setw(widthTime + 3) << "Average time";
    std::cout << std::setw(widthPercent) << "% of total";
    std::cout << std::endl;

    summary(0, total, nameWidth, depth);
};
