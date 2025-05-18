#pragma once
#include <string>
#include <map>
#include <vector>

bool save_pdb_to_file(const std::string& filename, const std::map<std::vector<int>, unsigned char>& pdb);
bool load_pdb_from_file(const std::string& filename, std::map<std::vector<int>, unsigned char>& pdb, int pattern_size);
