#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;


static int findIndex(std::vector<Point2f>& points, Point2f &point);
static void writeDelaunay(Subdiv2D& subdiv, std::vector<Point2f>& points, const std::string &filename);

int main() {
    // Create a vector of points.
    std::vector<Point2f> points;

    // Read in the points from a text file
    std::string pointsFilename("../data/images/smiling-man-delaunay.txt");
    std::ifstream ifs(pointsFilename);
    int x, y;
    while(ifs >> x >> y)
    {
        points.emplace_back(x,y);
    }

    std::cout << "Reading file " << pointsFilename << std::endl;

    // Find bounding box enclosing the points.
    Rect rect = boundingRect(points);

    // Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);

    // Insert points into subdiv
    for(auto & point : points)
    {
        subdiv.insert(point);
    }

    // Output filename
    std::string outputFileName("smiling-man-delaunay.tri");
    

    // Write delaunay triangles
    writeDelaunay(subdiv, points, outputFileName);

    std::cout << "Writing Delaunay triangles to " << outputFileName << std::endl;

    // Successful exit
    return 0;
}

// Write delaunay triangles to file
static void writeDelaunay(Subdiv2D& subdiv, std::vector<Point2f>& points, const std::string &filename)
{

    // Open file for writing
    std::ofstream ofs;
    ofs.open(filename);

    // Obtain the list of triangles.
    // Each triangle is stored as vector of 6 coordinates
    // (x0, y0, x1, y1, x2, y2)
    std::vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    // Will convert triangle representation to three vertices
    std::vector<Point2f> vertices(3);

    // Loop over all triangles
    for(auto t : triangleList)
    {
        // Obtain current triangle
        // Extract vertices of current triangle
        vertices[0] = Point2f(t[0], t[1]);
        vertices[1] = Point2f(t[2], t[3]);
        vertices[2] = Point2f(t[4], t[5]);

        // Find indices of vertices in the points list
        // and save to file.

        ofs << findIndex(points, vertices[0]) << " "
            << findIndex(points, vertices[1]) << " "
            << findIndex(points, vertices[2]) << std::endl;

    }
    ofs.close();
}


// In a vector of points, find the index of point closest to input point.
static int findIndex(std::vector<Point2f>& points, Point2f &point)
{
    int minIndex = 0;
    double minDistance = norm(points[0] - point);
    for(int i = 1; i < points.size(); i++)
    {
        double distance = norm(points[i] - point);
        if( distance < minDistance )
        {
            minIndex = i;
            minDistance = distance;
        }

    }
    return minIndex;
}