#include "pdb_loader.hpp"
#include <fstream>

bool save_pdb_to_file(const std::string& filename, const std::map<std::vector<int>, unsigned char>& pdb) {
    std::ofstream outfile(filename, std::ios::binary | std::ios::trunc);
    if (!outfile.is_open()) return false;
    size_t num_entries = pdb.size();
    outfile.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));
    for (const auto& pair : pdb) {
        const std::vector<int>& key_vec = pair.first;
        size_t key_size = key_vec.size();
        outfile.write(reinterpret_cast<const char*>(&key_size), sizeof(key_size));
        outfile.write(reinterpret_cast<const char*>(key_vec.data()), key_size * sizeof(int));
        unsigned char value = pair.second;
        outfile.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    outfile.close();
    return outfile.good();
}

bool load_pdb_from_file(const std::string& filename, std::map<std::vector<int>, unsigned char>& pdb, int pattern_size) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile.is_open()) return false;
    pdb.clear();
    size_t num_entries;
    infile.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
    if (infile.fail()) return false;
    for (size_t i = 0; i < num_entries; ++i) {
        size_t key_size;
        infile.read(reinterpret_cast<char*>(&key_size), sizeof(key_size));
        if (infile.fail() || key_size != (size_t)pattern_size) { pdb.clear(); return false; }
        std::vector<int> key_vec(key_size);
        infile.read(reinterpret_cast<char*>(key_vec.data()), key_size * sizeof(int));
        if (infile.fail()) { pdb.clear(); return false; }
        unsigned char value;
        infile.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (infile.fail()) { pdb.clear(); return false; }
        pdb[key_vec] = value;
    }
    infile.close();
    return !pdb.empty() && pdb.size() == num_entries;
}
