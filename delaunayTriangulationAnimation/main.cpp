#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

// Draw a point on an image using a specified color
static void drawPoint( Mat& img, const Point2f& fp, const Scalar& color )
{
    circle( img, fp, 2, color, FILLED, LINE_AA, 0 );
}


// Draw delaunay triangles
static void drawDelaunay( Mat& img, Subdiv2D& subdiv, const Scalar& delaunayColor )
{
    // Obtain the list of triangles.
    // Each triangle is stored as vector of 6 coordinates
    // (x0, y0, x1, y1, x2, y2)
    std::vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    // Will convert triangle representation to three vertices
    std::vector<Point> vertices(3);

    // Get size of the image
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);

    for(auto t : triangleList)
    {
        // Get current triangle
        // Convert triangle to vertices
        vertices[0] = Point(cvRound(t[0]), cvRound(t[1]));
        vertices[1] = Point(cvRound(t[2]), cvRound(t[3]));
        vertices[2] = Point(cvRound(t[4]), cvRound(t[5]));

        // Draw triangles that are completely inside the image.
        if ( rect.contains(vertices[0]) && rect.contains(vertices[1]) && rect.contains(vertices[2]))
        {
            line(img, vertices[0], vertices[1], delaunayColor, 1, LINE_AA, 0);
            line(img, vertices[1], vertices[2], delaunayColor, 1, LINE_AA, 0);
            line(img, vertices[2], vertices[0], delaunayColor, 1, LINE_AA, 0);
        }
    }
}

//Draw voronoi diagrams
static void drawVoronoi( Mat& img, Subdiv2D& subdiv )
{
    // Vector of voronoi facets.
    std::vector<std::vector<Point2f> > facets;

    // Voronoi centers
    std::vector<Point2f> centers;

    // Get facets and centers
    subdiv.getVoronoiFacetList(std::vector<int>(), facets, centers);

    // Variable for the ith facet used by fillConvexPoly
    std::vector<Point> ifacet;

    // Variable for the ith facet used by polylines.
    std::vector<std::vector<Point> > ifacets(1);

    for( size_t i = 0; i < facets.size(); i++ )
    {
        // Extract ith facet
        ifacet.resize(facets[i].size());
        for( size_t j = 0; j < facets[i].size(); j++ ) {
            ifacet[j] = facets[i][j];
        }

        // Generate random color
        Scalar color;
        color[0] = rand() & 255;
        color[1] = rand() & 255;
        color[2] = rand() & 255;

        // Fill facet with a random color
        fillConvexPoly(img, ifacet, color, 8, 0);

        // Draw facet boundary
        ifacets[0] = ifacet;
        polylines(img, ifacets, true, Scalar(), 1, LINE_AA, 0);

        // Draw centers.
        circle(img, centers[i], 3, Scalar(), FILLED, LINE_AA, 0);
    }
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

// Draw delaunay triangles
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


int main()
{

    // Define window names
    std::string win = "Delaunay Triangulation & Voronoi Diagram";

    // Define colors for drawing.
    Scalar delaunayColor(255,255,255), pointsColor(0, 0, 255);

    // Read in the image.
    Mat img = imread("../data/images/smiling-man.jpg");

    // Rectangle to be used with Subdiv2D
    Size size = img.size();
    Rect rect(0, 0, size.width, size.height);

    // Create an instance of Subdiv2D
    Subdiv2D subdiv(rect);

    // Create a vector of points.
    std::vector<Point2f> points;

    // Read in the points from a text file
    std::ifstream ifs("../data/images/smiling-man-delaunay.txt");
    int x, y;
    while(ifs >> x >> y)
    {
        points.emplace_back(x,y);
    }

    // Image for displaying Delaunay Triangulation
    Mat imgDelaunay;

    // Image for displaying Voronoi Diagram.
    Mat imgVoronoi = Mat::zeros(img.rows, img.cols, CV_8UC3);

    // Final side-by-side display.
    Mat imgDisplay;

    // Insert points into subdiv and animate
    for( auto it = points.begin(); it != points.end(); it++)
    {
        subdiv.insert(*it);

        imgDelaunay = img.clone();
        imgVoronoi = cv::Scalar(0,0,0);

        // Draw delaunay triangles
        drawDelaunay( imgDelaunay, subdiv, delaunayColor );

        // Draw points
        for(auto & point : points)
        {
            drawPoint(imgDelaunay, point, pointsColor);
        }

        // Draw voronoi map
        drawVoronoi(imgVoronoi, subdiv);

        hconcat(imgDelaunay, imgVoronoi, imgDisplay);
        imshow(win, imgDisplay);
        waitKey(100);
    }

    // Write delaunay triangles
    writeDelaunay(subdiv, points, "smiling-man-delaunay.tri");

    // Hold display after animation
    waitKey(0);

    // Successful exit
    return 0;
}

