#ifndef _HELPERS
#define _HELPERS

#include <iostream>
#include <filesystem>
#include <map>
#include <chrono>
#include <ctime>  
#include <boost/algorithm/string.hpp>

typedef boost::property_tree::ptree ConfigParams;

#define TIMESTAMP_STR_LENGTH 50 
#define LOG_OUT(X) { \
		std::time_t timestamp = std::time(nullptr); \
		char buf[TIMESTAMP_STR_LENGTH] = {}; \
		ctime_s(buf, TIMESTAMP_STR_LENGTH, &timestamp); \
		std::string s_buf = buf; \
		boost::algorithm::trim_right(s_buf); \
		std::cout << s_buf << " - " << X << std::endl; \
	}

#define LOG_DURATION(A, B) { \
	std::chrono::duration<double, std::milli> time_span = B - A; \
	LOG_OUT(std::format("Completed in {} milliseconds.", time_span.count())); \
}

#define EXEC_TIMED(X) { \
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now(); \
	X; \
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now(); \
	LOG_DURATION(t1, t2); \
}

std::string getFileParameter(const std::filesystem::path& configFilePath,
	const boost::property_tree::ptree& configParams,
	const std::string& key)
{
	auto fileParameter = configParams.get<std::filesystem::path>(key);
	if (fileParameter.is_relative())
	{
		auto configFileFolder = configFilePath.parent_path();
		return configFileFolder.append(configParams.get<std::string>(key)).string();
	}
	return fileParameter.string();
}

template<typename T>
T ReadEnumFromConfigParams(const ConfigParams& configParams,
	const std::map<std::string, T>& enum_map,
	const std::string& key,
	const std::string& defaultValue)
{
	auto stringFromConfig = configParams.get<std::string>(key, defaultValue);
	auto it = enum_map.find(stringFromConfig);
	if (it != enum_map.end())
	{
		return enum_map.at(stringFromConfig);
	}
	else
	{
		std::cout << "Invalid value '" + stringFromConfig + "' for key '" + key + "' found in config file. Exiting" << std::endl;
		exit(-1);
	}
}

#endif