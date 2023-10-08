#pragma once

// Argument parsing related utilites. //

#include <boost/program_options.hpp>
#include <iomanip>
#include <iostream>
#include <range/v3/all.hpp>
#include <unordered_map>

namespace esn {

namespace po = boost::program_options;
namespace rg = ranges;
namespace rgv = ranges::views;

/// Parse generic command line arguments and conditionally select what other parameters
/// to parse.
///
/// Example:
/// \code
///     po::options_description generic_arg_desc{"Generic options"};
///     generic_arg_desc.add_options()
///       ("help", "Produce help message.")
///       ("object-type", po::value<std::string>(), "Object type, one of {student, house}.");
///
///     po::options_description student_arg_desc{"Student options"};
///     student_arg_desc.add_options()
///       ("age", po::value<int>(), "Student's age.");
///
///     po::options_description house_arg_desc{"House options"};
///     house_arg_desc.add_options()
///       ("height", po::value<int>(), "House height.");
///
///     po::variables_map args = parse_conditional(argc, argv, generic_arg_desc,
///                                                {{"object-type",
///                                                   {{"student", student_arg_desc},
///                                                    {"house", house_arg_desc}}}});
/// \endcode
po::variables_map parse_conditional(
  int argc,
  char* argv[],
  const po::options_description& generic,
  const std::unordered_map<std::string, std::unordered_map<std::string, po::options_description>>&
    conditional = {})
{
    // Parse only generic arguments.
    po::parsed_options parsed_generic =
      po::command_line_parser(argc, argv).options(generic).allow_unregistered().run();
    po::variables_map args;
    po::store(parsed_generic, args);

    // Traverse through all the conditional descriptions and add them if they match.
    using value_desc_map = std::unordered_map<std::string, po::options_description>;
    po::options_description all_arg_desc = generic;
    for (const std::pair<const std::string, value_desc_map>& cdesc : conditional) {
        for (const std::pair<const std::string, po::options_description>& vdesc : cdesc.second) {
            if (args.at(cdesc.first).as<std::string>() == vdesc.first) {
                all_arg_desc.add(vdesc.second);
            }
        }
    }

    // Parse all arguments.
    po::store(po::parse_command_line(argc, argv, all_arg_desc), args);
    po::notify(args);

    // Print help message if the option is present.
    if (args.count("help")) {
        std::cout << all_arg_desc << "\n";
        std::exit(1);
    }

    return args;
}

std::ostream& operator<<(std::ostream& out, const po::variables_map& m)
{
    for (auto& [k, v] : m) {
        if (typeid(std::string) == v.value().type()) {
            out << "--" << k << '=';
            out << v.as<std::string>();
        } else if (typeid(double) == v.value().type()) {
            out << "--" << k << '=';
            out << std::setprecision(std::numeric_limits<double>::max_digits10) << v.as<double>();
        } else if (typeid(bool) == v.value().type()) {
            out << "--" << k << '=';
            out << v.as<bool>();
        } else if (typeid(long) == v.value().type()) {
            out << "--" << k << '=';
            out << v.as<long>();
        } else if (typeid(int) == v.value().type()) {
            out << "--" << k << '=';
            out << v.as<int>();
        } else if (typeid(std::vector<std::string>) == v.value().type()) {
            for (const std::string& sv : v.as<std::vector<std::string>>()) {
                out << "--" << k << '=' << sv << " ";
            }
        } else if (typeid(std::vector<long>) == v.value().type()) {
            for (long sv : v.as<std::vector<long>>()) {
                out << "--" << k << '=' << sv << " ";
            }
        } else if (typeid(std::vector<double>) == v.value().type()) {
            for (double sv : v.as<std::vector<double>>()) {
                out << "--" << k << '='
                    << std::setprecision(std::numeric_limits<double>::max_digits10) << sv << " ";
            }
        } else {
            out << "[UNPRINTABLE]";
        }
        if (k != m.rbegin()->first) out << " \\\n";
    }
    return out;
}

}  // namespace esn
